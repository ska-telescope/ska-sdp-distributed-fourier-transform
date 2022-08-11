# pylint: disable=logging-fstring-interpolation,consider-using-f-string
# pylint: disable=unused-argument
"""
demo using api
"""
import itertools
import logging
import os

import dask
import dask.array
import dask.distributed
import numpy
from distributed import performance_report
from distributed.diagnostics import MemorySampler

from scripts.utils import get_and_write_transfer
from src.api import (
    FacetConfig,
    SubgridConfig,
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
)
from src.api_helper import check_facet, make_facet, sparse_mask
from src.fourier_transform.dask_wrapper import set_up_dask
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def demo_api(queue_size, fundamental_params, lru_forward, lru_backward):
    """demo for api

    :param queue_size: size of queue
    :param fundamental_params: fundamental swift config
    """

    def process_subgrid(subgrid_config, subgrid_task):
        """return self for test"""
        return subgrid_task

    sources = [(1, 1, 0)]
    swiftlyconfig = SwiftlyConfig(**fundamental_params)

    # Temporary use of compete indexes to create test data
    subgrid_mask_off_dict = {}
    for idx, off in enumerate(swiftlyconfig.distriFFT.subgrid_off):
        subgrid_mask_off_dict[off] = swiftlyconfig.base_arrays.subgrid_A[idx]

    facet_mask_off_dict = {}
    for idx, off in enumerate(swiftlyconfig.distriFFT.facet_off):
        facet_mask_off_dict[off] = swiftlyconfig.base_arrays.facet_B[idx]

    subgrid_config_list = [
        SubgridConfig(
            off0,
            off1,
            sparse_mask(subgrid_mask_off_dict[off0]),
            sparse_mask(subgrid_mask_off_dict[off1]),
        )
        for off0, off1 in itertools.product(
            swiftlyconfig.distriFFT.subgrid_off,
            swiftlyconfig.distriFFT.subgrid_off,
        )
    ]

    facets_config_list = [
        FacetConfig(
            off0,
            off1,
            sparse_mask(facet_mask_off_dict[off0]),
            sparse_mask(facet_mask_off_dict[off1]),
        )
        for off0, off1 in itertools.product(
            swiftlyconfig.distriFFT.facet_off,
            swiftlyconfig.distriFFT.facet_off,
        )
    ]

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

    for subgrid_config in subgrid_config_list:
        subgrid_task = fwd.get_subgrid_task(subgrid_config)
        new_subgrid_task = dask.delayed(process_subgrid)(
            subgrid_config, subgrid_task
        )
        bwd.add_new_subgrid_task(subgrid_config, new_subgrid_task)

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
            demo_api(
                args.queue_size,
                SWIFT_CONFIGS[config_key],
                args.lru_forward,
                args.lru_backward,
            )

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
    dfft_parser.add_argument(
        "--lru_forward",
        type=int,
        default=1,
        help="max columns pin NMBF_BFs",
    )
    dfft_parser.add_argument(
        "--lru_backward",
        type=int,
        default=1,
        help="max columns pin NAF_MNAFs",
    )
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
