# pylint: disable=logging-fstring-interpolation
"""
demo using api
"""

import itertools
import logging
import os
import sys

import dask
import dask.array
import dask.distributed
import numpy

from src.api import (
    FacetConfig,
    SubgridConfig,
    check_facet,
    check_residual,
    check_subgrid,
    make_facet,
    make_subgrid,
    swiftly_backward,
    swiftly_forward,
    swiftly_major,
)
from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def demo_swiftly_forward(fundamental_params):
    """demo the use of swiftly_forward"""
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    sources = [(1, 1, 0)]
    facet_2 = [
        [
            dask.delayed(make_facet)(
                distr_fft.N,
                distr_fft.yB_s,
                distr_fft.facet_off[j0],
                base_arrays.facet_B[j0],
                distr_fft.facet_off[j1],
                base_arrays.facet_B[j1],
                sources,
            )
            or j1 in range(distr_fft.nfacet)
        ]
        for j0 in range(distr_fft.nfacet)
    ]

    facets_config_list = [
        FacetConfig(j0, j1, None, None, None)
        for j0, j1 in list(
            itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet))
        )
    ]

    subgrid_config_list = [
        SubgridConfig(i0, i1, None, None, None)
        for i0, i1 in list(
            itertools.product(
                range(distr_fft.nsubgrid), range(distr_fft.nsubgrid)
            )
        )
    ]

    for (i0, i1), subgrid_data_task in swiftly_forward(
        distr_fft,
        facets_config_list,
        facet_2,
        subgrid_config_list,
        base_arrays,
    ):
        check_task = dask.compute(
            dask.delayed(check_subgrid)(
                distr_fft.N,
                distr_fft.subgrid_off[i0],
                base_arrays.subgrid_A[i0],
                distr_fft.subgrid_off[i1],
                base_arrays.subgrid_A[i1],
                subgrid_data_task,
                sources,
            ),
            sync=True,
        )[0]
        log.info(f"finshed, {i0}, {i1}, subgrid_error, {check_task}")


def demo_swiftly_backward(fundamental_params):
    """demo backward

    :param fundamental_params: _description_
    """
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)
    sources = [(1, 1, 0)]
    subgrid_2 = [
        [
            dask.delayed(make_subgrid)(
                distr_fft.N,
                distr_fft.xA_size,
                distr_fft.subgrid_off[i0],
                base_arrays.subgrid_A[i0],
                distr_fft.subgrid_off[i1],
                base_arrays.subgrid_A[i1],
                sources,
            )
            for i0 in range(distr_fft.nsubgrid)
        ]
        for i1 in range(distr_fft.nsubgrid)
    ]

    facets_config_list = [
        FacetConfig(j0, j1, None, None, None)
        for j0, j1 in list(
            itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet))
        )
    ]

    subgrid_config_list = [
        SubgridConfig(i0, i1, None, None, None)
        for i0, i1 in list(
            itertools.product(
                range(distr_fft.nsubgrid), range(distr_fft.nsubgrid)
            )
        )
        # for i0,i1 in [(1,4),(10,34),(30,30)]
    ]

    for (i0, i1), handle_tasks in swiftly_backward(
        distr_fft,
        facets_config_list,
        subgrid_2,
        subgrid_config_list,
        base_arrays,
    ):
        # just a i0 i1 task-checker
        if i0 != -1 and i1 != -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i1 done: {i0},{i1}")

        # i0 task-checker
        elif i0 != -1 and i1 == -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i0 done: {i0},{i1}")
        # facet tasks
        elif i0 == -1 and i1 == -1:
            facet_tasks = handle_tasks
            check_task = [
                dask.delayed(check_facet)(
                    distr_fft.N,
                    distr_fft.facet_off[fc.j0],
                    base_arrays.facet_B[fc.j0],
                    distr_fft.facet_off[fc.j1],
                    base_arrays.facet_B[fc.j1],
                    facet,
                    sources,
                )
                for fc, facet in zip(facets_config_list, facet_tasks)
            ]
            check_facet_res = dask.compute(check_task)
            log.info("Facet errors: %s", str(check_facet_res))


def demo_major(fundamental_params):
    """demo the use of swiftly_major"""
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    sources = [(1, 1, 0)]
    obs_subgrid_2 = [
        [
            dask.delayed(make_subgrid)(
                distr_fft.N,
                distr_fft.xA_size,
                distr_fft.subgrid_off[i0],
                base_arrays.subgrid_A[i0],
                distr_fft.subgrid_off[i1],
                base_arrays.subgrid_A[i1],
                sources,
            )
            for i0 in range(distr_fft.nsubgrid)
        ]
        for i1 in range(distr_fft.nsubgrid)
    ]

    skymodel_facet_2 = [
        [
            dask.delayed(make_facet)(
                distr_fft.N,
                distr_fft.yB_size,
                distr_fft.facet_off[j0],
                base_arrays.facet_B[j0],
                distr_fft.facet_off[j1],
                base_arrays.facet_B[j1],
                sources,
            )
            for j1 in range(distr_fft.nfacet)
        ]
        for j0 in range(distr_fft.nfacet)
    ]

    facets_config_list = [
        FacetConfig(j0, j1, None, None, None)
        for j0, j1 in list(
            itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet))
        )
    ]

    subgrid_config_list = [
        SubgridConfig(i0, i1, None, None, None)
        for i0, i1 in list(
            itertools.product(
                range(distr_fft.nsubgrid), range(distr_fft.nsubgrid)
            )
        )
    ]

    for (i0, i1), handle_tasks in swiftly_major(
        distr_fft,
        facets_config_list,
        skymodel_facet_2,
        obs_subgrid_2,
        subgrid_config_list,
        base_arrays,
    ):
        if i0 != -1 and i1 != -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i1 done: {i0},{i1}")

        # i0 task-checker
        elif i0 != -1 and i1 == -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i0 done: {i0},{i1}")
        # facet tasks
        elif i0 == -1 and i1 == -1:
            facet_tasks = handle_tasks
            check_task = [
                dask.delayed(check_residual)(
                    facet,
                )
                for fc, facet in zip(facets_config_list, facet_tasks)
            ]
            check_facet_res = dask.compute(check_task)
            log.info("residual Facet errors: %s", str(check_facet_res))


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

        if args.demo_mode == "forward":
            demo_swiftly_forward(SWIFT_CONFIGS[config_key])
        elif args.demo_mode == "backward":
            demo_swiftly_backward(SWIFT_CONFIGS[config_key])
        elif args.demo_mode == "major":
            demo_major(SWIFT_CONFIGS[config_key])
        else:
            raise ValueError(
                "Only supported forward, backward and major demo mode"
            )

        dask_client.restart()

    tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    dfft_parser.add_argument(
        "--demo_mode",
        type=str,
        default="backward",
        help="api demo mode, forward, backward or major",
    )
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
