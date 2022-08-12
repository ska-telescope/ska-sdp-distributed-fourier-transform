"""
End-to-end api test
"""

import logging
import random

import dask
import pytest

from src.api import (
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
    make_full_facet_cover,
    make_full_subgrid_cover,
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
    "queue_size,lru_forward,lru_backward,shuffle",
    [
        (100, 1, 1, True),
        (200, 1, 1, True),
        (100, 2, 1, True),
        (200, 2, 1, True),
        (100, 1, 2, True),
        (200, 1, 2, True),
        (100, 1, 1, False),
        (200, 1, 1, False),
        (100, 2, 1, False),
        (200, 2, 1, False),
        (100, 1, 2, False),
        (200, 1, 2, False),
    ],
)
def test_swiftly_api(queue_size, lru_forward, lru_backward, shuffle):
    """test major with api"""
    client = set_up_dask()

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)

    subgrid_config_list = make_full_subgrid_cover(swiftlyconfig)
    facets_config_list = make_full_facet_cover(swiftlyconfig)

    facet_tasks = [
        (
            facet_config,
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facet_config.off0,
                facet_config.mask0,
                facet_config.off1,
                facet_config.mask1,
                sources,
            ),
        )
        for facet_config in facets_config_list
    ]

    fwd = SwiftlyForward(swiftlyconfig, facet_tasks, lru_forward, queue_size)
    bwd = SwiftlyBackward(
        swiftlyconfig, facets_config_list, lru_backward, queue_size
    )
    if shuffle:

        random.shuffle(subgrid_config_list)

    for subgrid_config in subgrid_config_list:
        subgrid_task = fwd.get_subgrid_task(subgrid_config)

        bwd.add_new_subgrid_task(subgrid_config, subgrid_task)

        log.info(
            "process subgrid off0: %d, off1: %d",
            subgrid_config.off0,
            subgrid_config.off1,
        )

    new_facet_tasks = bwd.finish()

    # check
    check_task = [
        dask.delayed(check_facet)(
            swiftlyconfig.distriFFT.N,
            facet_config.off0,
            facet_config.mask0,
            facet_config.off1,
            facet_config.mask1,
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
