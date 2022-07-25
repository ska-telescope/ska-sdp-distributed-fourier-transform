# pylint: disable=logging-fstring-interpolation
"""
demo using api
"""

import logging
import os

import dask
import dask.array
import dask.distributed
import numpy

from src.api import (
    FacetConfig,
    SubgridConfig,
    SwiftlyConfig,
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


def demo_swiftly_forward(fundamental_params):
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
                facets_config_list[j0][j1].facet_mask0,
                facets_config_list[j0][j1].facet_off1,
                facets_config_list[j0][j1].facet_mask1,
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

    for subgrid_config, subgrid_data_task in swiftly_forward(
        swiftlyconfig,
        facets_config_list,
        facet_data,
        subgrid_config_list,
    ):
        check_task = dask.compute(
            dask.delayed(check_subgrid)(
                swiftlyconfig.distriFFT.N,
                subgrid_config.subgrid_off0,
                subgrid_config.subgrid_mask0,
                subgrid_config.subgrid_off1,
                subgrid_config.subgrid_mask1,
                subgrid_data_task,
                sources,
            ),
            sync=True,
        )[0]

        log.info(
            "finshed, %d, %d, subgrid_error, %e",
            subgrid_config.i0,
            subgrid_config.i1,
            check_task,
        )


def demo_swiftly_backward(fundamental_params):
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
                swiftlyconfig.distriFFT.xA_size,
                subgrid_config_list[i0][i1].subgrid_off0,
                subgrid_config_list[i0][i1].subgrid_mask0,
                subgrid_config_list[i0][i1].subgrid_off1,
                subgrid_config_list[i0][i1].subgrid_mask1,
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

    for subgrid_config, (i0, i1), handle_tasks in swiftly_backward(
        swiftlyconfig,
        facets_config_list,
        subgrid_data,
        subgrid_config_list,
    ):
        # just a i0 i1 task-checker
        if i0 != -1 and i1 != -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(
                f"check task i1 done: {subgrid_config.i0},{subgrid_config.i1}"
            )

        # i0 task-checker
        elif i0 != -1 and i1 == -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i0 done: {i0},{i1}")
        # facet tasks
        elif i0 == -1 and i1 == -1:
            facet_tasks = handle_tasks
            check_task = [
                [
                    dask.delayed(check_facet)(
                        swiftlyconfig.distriFFT.N,
                        facet_config.facet_off0,
                        facet_config.facet_mask0,
                        facet_config.facet_off1,
                        facet_config.facet_mask1,
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
                        "(%d,%d), Facet errors: %e",
                        facet_config.j0,
                        facet_config.j1,
                        check_facet_res[facet_config.j0][facet_config.j1],
                    )


def demo_major(fundamental_params):
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

    obs_subgrid_data = [
        [
            dask.delayed(make_subgrid)(
                swiftlyconfig.distriFFT.N,
                subgrid_config_list[i0][i1].xA_size,
                subgrid_config_list[i0][i1].subgrid_off0,
                subgrid_config_list[i0][i1].subgrid_mask0,
                subgrid_config_list[i0][i1].subgrid_off1,
                subgrid_config_list[i0][i1].subgrid_mask1,
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

    skymodel_facet_data = [
        [
            dask.delayed(make_facet)(
                swiftlyconfig.distriFFT.N,
                swiftlyconfig.distriFFT.yB_size,
                facets_config_list[j0][j1].facet_off0,
                facets_config_list[j0][j1].facet_mask0,
                facets_config_list[j0][j1].facet_off1,
                facets_config_list[j0][j1].facet_mask1,
                sources,
            )
            for j1 in range(swiftlyconfig.distriFFT.nfacet)
        ]
        for j0 in range(swiftlyconfig.distriFFT.nfacet)
    ]

    for subgrid_config, (i0, i1), handle_tasks in swiftly_major(
        swiftlyconfig,
        facets_config_list,
        skymodel_facet_data,
        subgrid_config_list,
        obs_subgrid_data,
    ):
        if i0 != -1 and i1 != -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(
                f"check task i1 done: {subgrid_config.i0},{subgrid_config.i1}"
            )

        # i0 task-checker
        elif i0 != -1 and i1 == -1:
            handles = dask.compute(handle_tasks, sync=False)
            dask.distributed.wait(handles)
            log.info(f"check task i0 done: {i0},{i1}")
        # facet tasks
        elif i0 == -1 and i1 == -1:
            facet_tasks = handle_tasks
            check_task = [
                [
                    dask.delayed(check_residual)(
                        facet_tasks[j0][j1],
                    )
                    for j1, facet_config in enumerate(facet_config_j0)
                ]
                for j0, facet_config_j0 in enumerate(facets_config_list)
            ]

            check_facet_res = dask.compute(check_task)[0]
            for facet_config_j0 in facets_config_list:
                for facet_config in facet_config_j0:

                    log.info(
                        "(%d,%d), residual errors: %e",
                        facet_config.j0,
                        facet_config.j1,
                        check_facet_res[facet_config.j0][facet_config.j1],
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
