# pylint: disable=logging-fstring-interpolation,consider-using-f-string
# pylint: disable=unused-argument
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
)
from src.api_helper import check_facet, make_facet
from src.fourier_transform.dask_wrapper import set_up_dask
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def demo_api(queue_size, fundamental_params):
    """demo for api

    :param queue_size: size of queue
    :param fundamental_params: fundamental swift config
    """

    client = Client.current()

    def process_subgrid(subgrid_config, subgrid_task):
        """return self for test"""
        return subgrid_task

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

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
            filename="api-%s-queue-%d.html" % (config_key, args.queue_size)
        ), mem_sampler.sample(
            "process", measure="process"
        ), mem_sampler.sample(
            "managed", measure="managed"
        ):
            demo_api(args.queue_size, SWIFT_CONFIGS[config_key])

        mem_sampler.to_pandas().to_csv(
            "mem-api-%s-queue-%d.csv" % (config_key, args.queue_size)
        )

        get_and_write_transfer(
            dask_client,
            f"api-{config_key}-queue-{args.queue_size}",
        )

        dask_client.restart()


if __name__ == "__main__":
    dfft_parser = cli_parser()
    dfft_parser.add_argument(
        "--queue_size",
        type=int,
        default=20,
        help="the size of queue",
    )
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
