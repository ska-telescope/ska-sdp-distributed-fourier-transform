"""
End-to-end api test
"""

import itertools
import logging

import dask
import pytest

from src.api import (
    FacetConfig,
    SubgridConfig,
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
    TaskQueue,
)
from src.api_helper import check_facet, make_facet
from src.fourier_transform.dask_wrapper import set_up_dask

log = logging.getLogger("fourier-logger")
log.setLevel(logging.WARNING)


TEST_PARAMS = {
    "W": 13.25,
    "fov": 0.75,
    "N": 1024,
    "Nx": 4,
    "yB_size": 256,
    "yN_size": 320,
    "yP_size": 512,
    "xA_size": 188,
    "xM_size": 256,
}


@pytest.mark.parametrize(
    "queue_size,lru_forward,lru_backward",
    [
        (100, 1, 1),
        (200, 1, 1),
        (100, 2, 1),
        (200, 2, 1),
        (100, 1, 2),
        (200, 1, 2),
    ],
)
def test_swiftly_api(queue_size, lru_forward, lru_backward):
    """test major with api"""
    client = set_up_dask()

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)

    subgrid_config_list = [
        SubgridConfig(off0, off1)
        for off0, off1 in itertools.product(
            swiftlyconfig.distriFFT.subgrid_off,
            swiftlyconfig.distriFFT.subgrid_off,
        )
    ]

    facets_config_list = [
        FacetConfig(off0, off1)
        for off0, off1 in itertools.product(
            swiftlyconfig.distriFFT.facet_off,
            swiftlyconfig.distriFFT.facet_off,
        )
    ]

    subgrid_mask_off_dict = {}
    for idx, off in enumerate(swiftlyconfig.distriFFT.subgrid_off):
        subgrid_mask_off_dict[off] = swiftlyconfig.base_arrays.subgrid_A[idx]

    facet_mask_off_dict = {}
    for idx, off in enumerate(swiftlyconfig.distriFFT.facet_off):
        facet_mask_off_dict[off] = swiftlyconfig.base_arrays.facet_B[idx]

    # just for demo we scatter mask in caller
    facet_mask_off_dict_scatter = {}
    for off, facet_mask in facet_mask_off_dict.items():
        facet_mask_off_dict_scatter[off] = client.scatter(
            facet_mask, broadcast=True
        )

    facet_tasks = [
        (
            facet_config,
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facet_config.off0,
                facet_mask_off_dict_scatter[facet_config.off0],
                facet_config.off1,
                facet_mask_off_dict_scatter[facet_config.off1],
                sources,
            ),
        )
        for facet_config in facets_config_list
    ]

    # Temporary use of full data i0 number
    lru_same_off0_backward = swiftlyconfig.distriFFT.nsubgrid

    fwd = SwiftlyForward(
        swiftlyconfig, facet_tasks, subgrid_mask_off_dict, lru_forward
    )
    bwd = SwiftlyBackward(
        swiftlyconfig,
        facets_config_list,
        facet_mask_off_dict,
        lru_backward,
        lru_same_off0_backward,
    )
    task_queue = TaskQueue(queue_size)
    for subgrid_config in subgrid_config_list:
        subgrid_task = fwd.get_subgrid_task(subgrid_config)
        task_finished = bwd.add_new_subgrid_task(subgrid_config, subgrid_task)
        task_queue.process(
            "task", (subgrid_config.off0, subgrid_config.off1), task_finished
        )
        task_queue.empty_done()
    task_queue.wait_all_done()

    new_facet_tasks = bwd.finish()

    # check
    check_task = [
        dask.delayed(check_facet)(
            swiftlyconfig.distriFFT.N,
            facet_config.off0,
            facet_mask_off_dict_scatter[facet_config.off0],
            facet_config.off1,
            facet_mask_off_dict_scatter[facet_config.off1],
            new_facet,
            sources,
        )
        for new_facet, facet_config in zip(new_facet_tasks, facets_config_list)
    ]

    facet_error = dask.compute(check_task)[0]
    for facet_config, error in zip(facets_config_list, facet_error):
        log.info(
            "error facet, off0/off1:%d/%d: %e",
            facet_config.off0,
            facet_config.off1,
            error,
        )
        assert error < 3e-10

    client.close()
