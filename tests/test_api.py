"""
End-to-end api test
"""


import logging

import dask
import pytest

from src.api import (
    FacetConfig,
    SubgridConfig,
    SwiftlyConfig,
    TaskQueue,
    swiftly_backward,
    swiftly_forward,
    swiftly_major,
)
from src.api_helper import (
    check_facet,
    check_residual,
    check_subgrid,
    make_facet,
    make_subgrid,
)
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
def test_swiftly_forward(queue_size):
    """test forward with api"""

    client = set_up_dask()

    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)
    facets_config_list = [
        [
            FacetConfig(j0, j1, **TEST_PARAMS)
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]
    sources = [(1, 1, 0)]
    facet_data = [
        [
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facets_config_list[j0][j1].facet_off0,
                dask.delayed(facets_config_list[j0][j1].facet_mask0),
                facets_config_list[j0][j1].facet_off1,
                dask.delayed(facets_config_list[j0][j1].facet_mask1),
                sources,
            )
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]
    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **TEST_PARAMS)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]
    task_queue = TaskQueue(queue_size)
    for msg, subgrid_config, (i0, i1), handle_tasks in swiftly_forward(
        client,
        swiftlyconfig,
        facets_config_list,
        facet_data,
        subgrid_config_list,
    ):
        check_task = dask.delayed(check_subgrid)(
            swiftlyconfig.distriFFT.N,
            subgrid_config.subgrid_off0,
            dask.delayed(subgrid_config.subgrid_mask0),
            subgrid_config.subgrid_off1,
            dask.delayed(subgrid_config.subgrid_mask1),
            handle_tasks,
            sources,
        )

        task_queue.process(msg, (i0, i1), [[check_task]])
        for _, task in task_queue.done_tasks:
            assert task.result() < 1e-15

        task_queue.empty_done()

    # forward without finish facet, need wait all done.
    task_queue.wait_all_done()
    for _, task in task_queue.done_tasks:
        assert task.result() < 1e-15
    client.close()


@pytest.mark.parametrize(
    "queue_size",
    [1, 2],
)
def test_swiftly_backward(queue_size):
    """test backward with api"""

    client = set_up_dask()

    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **TEST_PARAMS)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]
    sources = [(1, 512, 512)]
    subgrid_data = [
        [
            dask.delayed(make_subgrid)(
                swiftlyconfig.distriFFT.N,
                subgrid_config_list[i0][i1].xA_size,
                subgrid_config_list[i0][i1].subgrid_off0,
                dask.delayed(subgrid_config_list[i0][i1].subgrid_mask0),
                subgrid_config_list[i0][i1].subgrid_off1,
                dask.delayed(subgrid_config_list[i0][i1].subgrid_mask1),
                sources,
            )
            for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    facets_config_list = [
        [
            FacetConfig(j0, j1, **TEST_PARAMS)
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    task_queue = TaskQueue(queue_size)
    for msg, subgrid_config, (i0, i1), handle_tasks in swiftly_backward(
        client,
        swiftlyconfig,
        facets_config_list,
        subgrid_data,
        subgrid_config_list,
    ):
        # facet tasks
        if i0 == -1 and i1 == -1 and subgrid_config == -1:
            facet_tasks = handle_tasks
            check_task = [
                [
                    dask.delayed(check_facet)(
                        swiftlyconfig.distriFFT.N,
                        facet_config.facet_off0,
                        dask.delayed(facet_config.facet_mask0),
                        facet_config.facet_off1,
                        dask.delayed(facet_config.facet_mask1),
                        facet_tasks[j0][j1],
                        sources,
                    )
                    for j1, facet_config in enumerate(facet_config_j0)
                ]
                for j0, facet_config_j0 in enumerate(facets_config_list)
            ]
            check_facet_res = dask.compute(check_task)[0]
            for facet_config_j0 in facets_config_list:
                for facet_config in facet_config_j0:
                    assert (
                        check_facet_res[facet_config.j0][facet_config.j1]
                        < 1.675e-3
                    )
        else:
            task_queue.process(msg, (i0, i1), handle_tasks)
            task_queue.empty_done()
    task_queue.wait_all_done()
    client.close()


@pytest.mark.parametrize(
    "queue_size",
    [1, 2],
)
def test_swiftly_major(queue_size):
    """test major with api"""
    client = set_up_dask()

    swiftlyconfig = SwiftlyConfig(**TEST_PARAMS)

    sources = [(1, 1, 0)]

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **TEST_PARAMS)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    obs_subgrid_data = [
        [
            dask.delayed(make_subgrid)(
                swiftlyconfig.distriFFT.N,
                subgrid_config_list[i0][i1].xA_size,
                subgrid_config_list[i0][i1].subgrid_off0,
                dask.delayed(subgrid_config_list[i0][i1].subgrid_mask0),
                subgrid_config_list[i0][i1].subgrid_off1,
                dask.delayed(subgrid_config_list[i0][i1].subgrid_mask1),
                sources,
            )
            for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    facets_config_list = [
        [
            FacetConfig(j0, j1, **TEST_PARAMS)
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    skymodel_facet_data = [
        [
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facets_config_list[j0][j1].facet_off0,
                dask.delayed(facets_config_list[j0][j1].facet_mask0),
                facets_config_list[j0][j1].facet_off1,
                dask.delayed(facets_config_list[j0][j1].facet_mask1),
                sources,
            )
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    task_queue = TaskQueue(queue_size)
    for msg, subgrid_config, (i0, i1), handle_tasks in swiftly_major(
        client,
        swiftlyconfig,
        facets_config_list,
        skymodel_facet_data,
        subgrid_config_list,
        obs_subgrid_data,
    ):
        # facet tasks
        if i0 == -1 and i1 == -1 and subgrid_config == -1:
            facet_tasks = handle_tasks
            check_task = [
                [
                    dask.delayed(check_residual)(
                        facet_tasks[facet_config.j0][facet_config.j1],
                    )
                    for facet_config in facet_config_j0
                ]
                for facet_config_j0 in facets_config_list
            ]

            check_facet_res = dask.compute(check_task)[0]
            for facet_config_j0 in facets_config_list:
                for facet_config in facet_config_j0:
                    assert (
                        check_facet_res[facet_config.j0][facet_config.j1]
                        < 0.00573
                    )
        else:
            task_queue.process(msg, (i0, i1), handle_tasks)
            task_queue.empty_done()

    task_queue.wait_all_done()
    client.close()
