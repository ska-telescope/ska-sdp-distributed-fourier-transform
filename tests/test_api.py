"""
End-to-end api test
"""


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
    "queue_size",
    [1, 2],
)
def test_swiftly_api(queue_size):
    """test major with api"""
    client = set_up_dask()

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    facets_config_list = [
        [FacetConfig(j0, j1) for j1 in range(swiftlyconfig.distriFFT.nfacet)]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    facets_mask_task_list = [
        [
            (
                client.scatter(
                    swiftlyconfig.base_arrays.facet_B[j0], broadcast=True
                ),
                client.scatter(
                    swiftlyconfig.base_arrays.facet_B[j1], broadcast=True
                ),
            )
            for j0 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j1 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    facet_data = [
        [
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                swiftlyconfig.distriFFT.facet_off[j0],
                facets_mask_task_list[j0][j1][0],
                swiftlyconfig.distriFFT.facet_off[j1],
                facets_mask_task_list[j0][j1][1],
                sources,
            )
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    facet_tasks = [
        [
            (facets_config_list[j0][j1], facet_data[j0][j1])
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    subgrid_configs = []
    for i0_subgrid in subgrid_config_list:
        for subgrid in i0_subgrid:
            subgrid_configs.append(subgrid)

    fwd = SwiftlyForward(swiftlyconfig, facet_tasks)
    bwd = SwiftlyBackward(swiftlyconfig)
    task_queue = TaskQueue(queue_size)
    for subgrid_config in subgrid_configs:
        subgrid_task = fwd.get_subgrid_task(subgrid_config)
        handle_task = bwd.add_new_subgrid_task(subgrid_config, subgrid_task)
        task_queue.process(
            "hd", (subgrid_config.i0, subgrid_config.i1), handle_task
        )
        task_queue.empty_done()
        log.info(
            "process i0: %d, i1: %d", subgrid_config.i0, subgrid_config.i1
        )
    task_queue.wait_all_done()

    new_facet_tasks = bwd.finish()

    # check
    check_task = [
        [
            dask.delayed(check_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.facet_off[j0],
                facets_mask_task_list[j0][j1][0],
                swiftlyconfig.distriFFT.facet_off[j1],
                facets_mask_task_list[j0][j1][1],
                new_facet_tasks[j0][j1],
                sources,
            )
            for j1, facet_config in enumerate(facet_config_j0)
        ]
        for j0, facet_config_j0 in enumerate(facets_config_list)
    ]

    error = dask.compute(check_task)[0]
    for error_i0 in error:
        for error_i1 in error_i0:
            log.info("error: %e", error_i1)
            assert error_i1 < 1e-8

    client.close()
