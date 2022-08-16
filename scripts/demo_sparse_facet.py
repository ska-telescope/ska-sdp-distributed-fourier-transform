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
from distributed import performance_report
from distributed.diagnostics import MemorySampler

from scripts.utils import get_and_write_transfer
from src.api import (
    FacetConfig,
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
    make_full_subgrid_cover,
)
from src.api_helper import check_facet, make_facet
from src.fourier_transform.dask_wrapper import set_up_dask
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def make_sparse_facet_cover_from_list(off_list, mask_list):
    """make facet_config from off and mask list

    :param off_list: off_list
    :param mask_list: mask_list
    :return: config_list
    """

    config_list = []
    for (off0, off1), (mask0, mask1) in zip(off_list, mask_list):
        config_list.append(FacetConfig(off0, off1, mask0, mask1))

    return config_list


def make_demo_sparse_off(swiftlyconfig):
    """only use 4x4 facet and get 7 facet on centre
    32k[1]-n10k-1k"""

    N = swiftlyconfig.distriFFT.N
    yB = swiftlyconfig.distriFFT.yB_size

    off_list = []
    # up
    off1 = yB
    off_list.append((yB // 2, off1))
    off_list.append((N - yB // 2, off1))

    # centre
    off1 = 0
    off_list.append((0, off1))
    off_list.append((yB, off1))
    off_list.append((N - yB, off1))

    # down
    off1 = N - yB // 2
    off_list.append((yB // 2, off1))
    off_list.append((N - yB // 2, off1))

    mask_list = [
        ([[slice(None, None, None)], yB], [[slice(None, None, None)], yB])
        for _ in off_list
    ]
    return off_list, mask_list


def demo_api(queue_size, fundamental_params, lru_forward, lru_backward):
    """demo for api

    :param queue_size: size of queue
    :param fundamental_params: fundamental swift config
    """

    def process_subgrid(subgrid_config, subgrid_task):
        """return self for test"""
        return subgrid_task

    swiftlyconfig = SwiftlyConfig(**fundamental_params)
    sources = [(1, 1, 0)]

    subgrid_config_list = make_full_subgrid_cover(swiftlyconfig)
    # facets_config_list = make_full_facet_cover(swiftlyconfig)

    # demo sparse facet
    off_list, mask_list = make_demo_sparse_off(swiftlyconfig)
    facets_config_list = make_sparse_facet_cover_from_list(off_list, mask_list)

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
            "error facet, off0/off1: %d-%d: %e",
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
