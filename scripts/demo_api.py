# pylint: disable=logging-fstring-interpolation,consider-using-f-string
"""
demo using api
"""

import logging
import os

import dask
import dask.array
import dask.distributed
import numpy
from distributed import Client, performance_report
from distributed.diagnostics import MemorySampler

from scripts.utils import get_and_write_transfer
from src.api import (
    FacetConfig,
    SubgridConfig,
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
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
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def demo_swiftly_forward(client, queue_size, fundamental_params):
    """demo the use of swiftly_forward"""
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

    facets_config_list = [
        [
            FacetConfig(j0, j1, **fundamental_params)
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
            SubgridConfig(i0, i1, **fundamental_params)
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
        for meta, task in task_queue.done_tasks:
            log.info(
                "%s,check done: %d,%d, handle: (%d/%d), subgrid error: %e",
                meta[2],
                meta[3][0],
                meta[3][1],
                meta[0],
                meta[1],
                task.result(),
            )
        task_queue.empty_done()

    # forward without finish facet, need wait all done.
    task_queue.wait_all_done()
    for meta, task in task_queue.done_tasks:
        log.info(
            "%s,check done: %d,%d, handle: (%d/%d), subgrid error: %e",
            meta[2],
            meta[3][0],
            meta[3][1],
            meta[0],
            meta[1],
            task.result(),
        )


def demo_swiftly_backward(client, queue_size, fundamental_params):
    """demo backward

    :param fundamental_params: _description_
    """
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **fundamental_params)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]
    sources = [(1, 1, 0)]
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
            FacetConfig(j0, j1, **fundamental_params)
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
                    log.info(
                        "%s,(%d,%d), Facet errors: %e",
                        msg,
                        facet_config.j0,
                        facet_config.j1,
                        check_facet_res[facet_config.j0][facet_config.j1],
                    )
        else:
            task_queue.process(msg, (i0, i1), handle_tasks)
            for meta, _ in task_queue.done_tasks:
                # i1 task-checker
                if meta[3][0] != -1 and meta[3][1] != -1:
                    log.info(
                        "%s,check task i1 done: %d,%d, handle task: (%d/%d)",
                        meta[2],
                        meta[3][0],
                        meta[3][1],
                        meta[0],
                        meta[1],
                    )
                # i0 task-checker
                elif meta[3][0] != -1 and meta[3][1] == -1:
                    log.info(
                        "%s,check task i0 done: %d,%d, handle task: (%d/%d)",
                        meta[2],
                        meta[3][0],
                        meta[3][1],
                        meta[0],
                        meta[1],
                    )
            task_queue.empty_done()
    task_queue.wait_all_done()


def demo_major(client, queue_size, fundamental_params):
    """demo the use of swiftly_major"""
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

    sources = [(1, 1, 0)]

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **fundamental_params)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    subgrid_mask_task_list = [
        [
            (
                client.scatter(subgrid_config.subgrid_mask0, broadcast=True),
                client.scatter(subgrid_config.subgrid_mask1, broadcast=True),
            )
            for subgrid_config in subgrid_config_i0
        ]
        for subgrid_config_i0 in subgrid_config_list
    ]

    obs_subgrid_data = [
        [
            dask.delayed(make_subgrid)(
                swiftlyconfig.distriFFT.N,
                subgrid_config_list[i0][i1].xA_size,
                subgrid_config_list[i0][i1].subgrid_off0,
                subgrid_mask_task_list[i0][i1][0],
                # dask.delayed(subgrid_config_list[i0][i1].subgrid_mask0),
                subgrid_config_list[i0][i1].subgrid_off1,
                subgrid_mask_task_list[i0][i1][1],
                # dask.delayed(subgrid_config_list[i0][i1].subgrid_mask1),
                sources,
            )
            for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    facets_config_list = [
        [
            FacetConfig(j0, j1, **fundamental_params)
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    facets_mask_task_list = [
        [
            (
                client.scatter(facets_config.facet_mask0, broadcast=True),
                client.scatter(facets_config.facet_mask1, broadcast=True),
            )
            for facets_config in facets_config_j0
        ]
        for facets_config_j0 in facets_config_list
    ]

    skymodel_facet_data = [
        [
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facets_config_list[j0][j1].facet_off0,
                facets_mask_task_list[j0][j1][0],
                # dask.delayed(facets_config_list[j0][j1].facet_mask0),
                facets_config_list[j0][j1].facet_off1,
                facets_mask_task_list[j0][j1][1],
                # dask.delayed(facets_config_list[j0][j1].facet_mask1),
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
        facets_mask_task_list,
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
                    log.info(
                        "%s,(%d,%d), residual errors: %e",
                        msg,
                        facet_config.j0,
                        facet_config.j1,
                        check_facet_res[facet_config.j0][facet_config.j1],
                    )
        else:
            task_queue.process(msg, (i0, i1), handle_tasks)
            for meta, _ in task_queue.done_tasks:
                # i1 task-checker
                if meta[3][0] != -1 and meta[3][1] != -1:
                    log.info(
                        "%s,check task i1 done: %d,%d, handle task: (%d/%d)",
                        meta[2],
                        meta[3][0],
                        meta[3][1],
                        meta[0],
                        meta[1],
                    )
                # i0 task-checker
                elif meta[3][0] != -1 and meta[3][1] == -1:
                    log.info(
                        "%s,check task i0 done: %d,%d, handle task: (%d/%d)",
                        meta[2],
                        meta[3][0],
                        meta[3][1],
                        meta[0],
                        meta[1],
                    )
            task_queue.empty_done()

    task_queue.wait_all_done()


def demo_api_new(fundamental_params):
    client = Client.current()

    # return self to test
    def process_subgrid(subgrid_config, subgrid_task):
        return subgrid_task

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

    subgrid_config_list = [
        [
            SubgridConfig(i0, i1, **fundamental_params)
            for i1 in range(swiftlyconfig.distriFFT.nsubgrid)
        ]
        for i0 in range(swiftlyconfig.distriFFT.nsubgrid)
    ]

    facets_config_list = [
        [
            FacetConfig(j0, j1, **fundamental_params)
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
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
    task_queue = TaskQueue(9)
    for subgrid_config in subgrid_configs:
        subgrid_task = fwd.get_subgrid_task(subgrid_config)
        new_subgrid_task = dask.delayed(process_subgrid)(
            subgrid_config, subgrid_task
        )
        handle_task = bwd.add_new_subgrid_task(
            subgrid_config, new_subgrid_task
        )
        task_queue.process(
            "hd", (subgrid_config.i0, subgrid_config.i1), handle_task
        )
        task_queue.empty_done()
        print(subgrid_config.i0, subgrid_config.i1)
    task_queue.wait_all_done()

    new_facet_tasks = bwd.finish()

    # check
    check_task = [
        [
            dask.delayed(check_facet)(
                swiftlyconfig.distriFFT.N,
                facet_config.facet_off0,
                facets_mask_task_list[j0][j1][0],
                facet_config.facet_off1,
                facets_mask_task_list[j0][j1][1],
                new_facet_tasks[j0][j1],
                sources,
            )
            for j1, facet_config in enumerate(facet_config_j0)
        ]
        for j0, facet_config_j0 in enumerate(facets_config_list)
    ]

    res = dask.compute(check_task)
    print(res)


def main(args):
    """main function"""
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    dask_client = set_up_dask(scheduler_address=scheduler)

    swift_config_keys = args.swift_config.split(",")
    for c in swift_config_keys:
        try:
            SWIFT_CONFIGS[c]
        except KeyError as error:
            raise KeyError(
                f"Provided argument ({c}) does not match any swift "
                f"configuration keys. Please consult src/swift_configs.py "
                f"for available options."
            ) from error

    for config_key in swift_config_keys:
        log.info("Running for swift-config: %s", config_key)
        mem_sampler = MemorySampler()

        with performance_report(
            filename="mode-%s-%s-queue-%d.html"
            % (args.demo_mode, config_key, args.queue_size)
        ), mem_sampler.sample(
            "process", measure="process"
        ), mem_sampler.sample(
            "managed", measure="managed"
        ):
            if args.demo_mode == "forward":
                demo_swiftly_forward(
                    dask_client, args.queue_size, SWIFT_CONFIGS[config_key]
                )
            elif args.demo_mode == "backward":
                demo_swiftly_backward(
                    dask_client, args.queue_size, SWIFT_CONFIGS[config_key]
                )
            elif args.demo_mode == "major":
                demo_major(
                    dask_client, args.queue_size, SWIFT_CONFIGS[config_key]
                )
            elif args.demo_mode == "new":
                demo_api_new(SWIFT_CONFIGS[config_key])
            else:
                raise ValueError(
                    "Only supported forward, backward and major demo mode"
                )

        mem_sampler.to_pandas().to_csv(
            "mem-mode-%s-%s-queue-%d.csv"
            % (args.demo_mode, config_key, args.queue_size)
        )

        get_and_write_transfer(
            dask_client,
            f"mode-{args.demo_mode}-{config_key}-queue-{args.queue_size}",
        )

        dask_client.restart()


if __name__ == "__main__":
    dfft_parser = cli_parser()
    dfft_parser.add_argument(
        "--demo_mode",
        type=str,
        default="backward",
        help="api demo mode, forward, backward or major",
    )
    dfft_parser.add_argument(
        "--queue_size",
        type=int,
        default=20,
        help="the size of queue",
    )
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
