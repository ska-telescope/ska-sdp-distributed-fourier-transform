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
from utils import cli_parser, get_and_write_transfer

from ska_sdp_exec_swiftly import (
    SWIFT_CONFIGS,
    FacetConfig,
    SwiftlyBackward,
    SwiftlyConfig,
    SwiftlyForward,
    check_facet,
    make_facet,
    make_full_subgrid_cover,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def calc_off0_per_row(facet_size, nfacet, N):
    """
    calculate off0 list in a row base on nfacet

    @param facet_size: facet_size
    @param nfacet: number of facet
    @param N: N pixelx
    @return: off_list

    """
    off_list = []
    if nfacet % 2 == 0:
        offset0 = facet_size // 2
        for i in range(nfacet // 2):
            off_right = offset0 + i * facet_size
            off_left = N - off_right
            off_list.append(off_right)
            off_list.append(off_left)
    else:
        offset0 = 0
        off_list.append(offset0)
        for i in range(1, (nfacet + 1) // 2):
            off_right = offset0 + i * facet_size
            off_left = N - off_right
            off_list.append(off_right)
            off_list.append(off_left)
    return off_list


def calc_nfacet_and_off1(facet_size, fov_size, N):
    """
    calculate nfacet and off1

    @param facet_size: facet size
    @param fov_size: the facet size we wanted
    @param N: N pixels
    @return: off1_nfacet_list
    """
    n_rows = int(numpy.ceil(fov_size / facet_size))
    off1_nfacet_list = []
    if n_rows % 2 == 0:
        offset0 = facet_size // 2
        for i in range(n_rows // 2):
            off1_up = offset0 + i * facet_size
            off1_down = N - off1_up
            if i == 0:
                largest = fov_size
            else:
                largest = 2 * numpy.sqrt(
                    (fov_size / 2) ** 2 - (off1_up - facet_size / 2) ** 2
                )
            nfacet = int(numpy.ceil(largest / facet_size))
            off1_nfacet_list.append((nfacet, off1_up))
            off1_nfacet_list.append((nfacet, off1_down))

    else:
        offset0 = 0
        off1_nfacet_list.append((n_rows, offset0))

        for i in range(1, (n_rows + 1) // 2):
            off1_up = offset0 + i * facet_size
            off1_down = N - off1_up

            largest = 2 * numpy.sqrt(
                (fov_size / 2) ** 2 - (off1_up - facet_size / 2) ** 2
            )
            nfacet = int(numpy.ceil(largest / facet_size))
            off1_nfacet_list.append((nfacet, off1_up))
            off1_nfacet_list.append((nfacet, off1_down))
    return off1_nfacet_list


def fov_sparse_cover_off_mask(swiftlyconfig, ifov_pixel, x=0, y=0):
    """calculate fov sparse cover off and mask list

    @param swiftlyconfig: Switftlyconfig
    @param ifov_pixel: the
    @param x:  Fov offset in x axis
    @param y: Fov offset in y axis
    @return: off0_off1_list, mask_list
    """
    N = swiftlyconfig.distriFFT.N
    facet_size = swiftlyconfig.distriFFT.yB_size
    off0_off1_list = []
    for nfacet, off1 in calc_nfacet_and_off1(facet_size, ifov_pixel, N):

        for off0 in calc_off0_per_row(facet_size, nfacet, N):

            off0_off1_list.append((off0 + x, off1 + y))
    mask_list = [
        (
            [[slice(None, None, None)], facet_size],
            [[slice(None, None, None)], facet_size],
        )
        for _ in off0_off1_list
    ]

    facet_off_step = swiftlyconfig.distriFFT.facet_off_step
    for off0, off1 in off0_off1_list:
        if off0 % facet_off_step != 0 or off1 % facet_off_step != 0:
            raise ValueError("Can't not support offset % (N//Nx) != 0")

    return off0_off1_list, mask_list


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
    """
    Generate all necessary sparse facet parameters,
    supporting configurations if Fov assigned less than the original Fov
    """
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
    off1 = N - yB
    off_list.append((yB // 2, off1))
    off_list.append((N - yB // 2, off1))

    mask_list = [
        ([[slice(None, None, None)], yB], [[slice(None, None, None)], yB])
        for _ in off_list
    ]
    return off_list, mask_list


def demo_api(
    queue_size, fundamental_params, lru_forward, lru_backward, source_count
):
    """demo for api

    :param queue_size: size of queue
    :param fundamental_params: fundamental swift config
    """

    def process_subgrid(subgrid_config, subgrid_task):
        """return self for test"""
        return subgrid_task

    swiftlyconfig = SwiftlyConfig(**fundamental_params)
    sources = [(1, i + 1, i) for i in range(source_count)]
    print(sources)

    subgrid_config_list = make_full_subgrid_cover(swiftlyconfig)
    # facets_config_list = make_full_facet_cover(swiftlyconfig)

    # demo sparse facet
    ifov_pixel = int(2.12 * swiftlyconfig.distriFFT.yB_size)

    off_list, mask_list = fov_sparse_cover_off_mask(swiftlyconfig, ifov_pixel)
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
    dask_client = dask.distributed.Client(scheduler)
    log.info(dask_client.dashboard_link)

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
                args.source_number,
            )

        mem_sampler.to_pandas().to_csv(
            "mem-api-%s-queue-%d.csv" % (config_key, args.queue_size)
        )

        get_and_write_transfer(
            dask_client,
            f"api-{config_key}-queue-{args.queue_size}",
        )

        dask_client.restart()


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

if __name__ == "__main__":
    logging.basicConfig()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
