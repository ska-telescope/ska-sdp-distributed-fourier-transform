# pylint: disable=too-many-arguments, unused-argument
# pylint: disable=logging-fstring-interpolation

"""
    Test script for memory consumption (ORC-1247)
    Created by Feng Wang
    Currently it only tests the prepare facet step

"""
import itertools
import logging
import os
import sys

import dask
import numpy
from dask.distributed import wait
from distributed import performance_report
from distributed.diagnostics import MemorySampler
from utils import sum_and_finish_subgrid, wait_for_tasks

from ska_sdp_exec_swiftly import (
    SWIFT_CONFIGS,
    BaseArrays,
    StreamingDistributedFFT,
    cli_parser,
    make_subgrid_and_facet,
    set_up_dask,
    tear_down_dask,
)
from ska_sdp_exec_swiftly.utils import generate_input_data

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def run_facet_to_subgrid(fundamental_params, use_dask=True):
    """
    A variation of the execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    Do not use HDF5 for simplification, and always uses Dask.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param use_dask: run function with dask.delayd or not?

    :return: ms_df: memory information
    """

    distr_fft = StreamingDistributedFFT(**fundamental_params)

    base_arrays = BaseArrays(**fundamental_params)

    # Generate the simplest input data
    G_2, FG_2 = generate_input_data(distr_fft, source_count=0)

    log.info("------------------------------------------")

    G_2 = dask.delayed(G_2)
    FG_2 = dask.delayed(FG_2)

    _, facet_2 = make_subgrid_and_facet(
        G_2,
        FG_2,
        base_arrays,
        dims=2,
        use_dask=use_dask,
    )

    facet_2 = dask.persist(facet_2)[0]
    # Wait for all tasks of facet_2 to finish, then continue
    wait(facet_2)

    # Calculate expected memory usage
    max_work_tasks = 9
    cpx_size = numpy.dtype(complex).itemsize

    log.info(" == Expected memory usage:")
    nfacet2 = distr_fft.nfacet**2
    max_work_columns = (
        1 + (max_work_tasks + distr_fft.nsubgrid - 1) // distr_fft.nsubgrid
    )
    BF_F_size = cpx_size * nfacet2 * distr_fft.yB_size * distr_fft.yP_size
    NMBF_BF_size = (
        max_work_columns
        * cpx_size
        * nfacet2
        * distr_fft.yP_size
        * distr_fft.xM_yN_size
    )
    NMBF_NMBF_size = (
        max_work_tasks
        * cpx_size
        * nfacet2
        * distr_fft.xM_yN_size
        * distr_fft.xM_yN_size
    )
    sum_size = BF_F_size + NMBF_BF_size + NMBF_NMBF_size
    log.info("BF_F (facets): %.3f GB", BF_F_size / 1e9)
    log.info("NMBF_BF (subgrid columns):    %.3f     GB", NMBF_BF_size / 1e9)
    log.info(
        "NMBF_NMBF (subgrid contributions): %.3f GB", NMBF_NMBF_size / 1e9
    )
    log.info("Sum: %.3f GB", sum_size / 1e9)

    # List of all facet indices
    facet_ixs = list(
        itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet))
    )
    ms = MemorySampler()
    with ms.sample("NMBF_NMBF.process", measure="process"):
        with ms.sample("NMBF_NMBF.managed", measure="managed"):

            # ** Step 1: Facet preparation first axis

            # Generate BF_F terms (parallel over all facets)
            BF_F_tasks = dask.persist(
                [
                    distr_fft.prepare_facet(
                        facet_2[j0][j1],
                        base_arrays.Fb,
                        axis=0,
                        use_dask=use_dask,
                        nout=1,
                    )
                    for j0, j1 in facet_ixs
                ]
            )[0]
            # ** Step 2: Extraction first axis, preparation second axis
            # Job queue
            work_tasks = []
            # Sequentially go over subgrid columns
            for i0 in list(range(distr_fft.nsubgrid)):

                # Generate NMBF_BF terms (parallel over all facets)
                NMBF_BF_tasks = dask.persist(
                    [
                        distr_fft.prepare_facet(
                            distr_fft.extract_facet_contrib_to_subgrid(
                                BF_F,
                                distr_fft.subgrid_off[i0],
                                base_arrays.facet_m0_trunc,
                                base_arrays.Fn,
                                axis=0,
                                use_dask=use_dask,
                                nout=1,
                            ),
                            base_arrays.Fb,
                            axis=1,
                            use_dask=use_dask,
                            nout=1,
                        )
                        for (j0, j1), BF_F in zip(facet_ixs, BF_F_tasks)
                    ]
                )[0]
                # ** Step 3: Extraction second axis
                # Sequential go over individual subgrids
                # sleep_tasks = []
                for i1 in list(range(distr_fft.nsubgrid)):
                    # No space in work queue?
                    while len(work_tasks) >= max_work_tasks:
                        work_tasks = wait_for_tasks(
                            work_tasks, return_when="FIRST_COMPLETED"
                        )
                    # Generate NMBF_NMBF tasks (parallel over all facets)
                    NMBF_NMBF_tasks = [
                        distr_fft.extract_facet_contrib_to_subgrid(
                            NMBF_BF,
                            distr_fft.subgrid_off[i1],
                            base_arrays.facet_m0_trunc,
                            base_arrays.Fn,
                            axis=1,
                            use_dask=use_dask,
                            nout=1,
                        )
                        for (j0, j1), NMBF_BF in zip(facet_ixs, NMBF_BF_tasks)
                    ]
                    # ** Step 4: Generate subgrids
                    # As a single Dask task - no point in splitting this
                    work_tasks += dask.compute(
                        dask.delayed(sum_and_finish_subgrid)(
                            distr_fft,
                            base_arrays,
                            i0,
                            i1,
                            facet_ixs,
                            NMBF_NMBF_tasks,
                        ),
                        sync=False,
                    )
            # Finish tasks
            wait_for_tasks(work_tasks)

    ms_df = ms.to_pandas()
    return ms_df


def main(args):
    """
    Main function to run the Distributed FFT
    """
    dask_config = dask.config.get(
        "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_"
    )
    log.info(f"MALLOC_TRIM_THRESHOLD_:{dask_config}")
    # Fixing seed of numpy random
    numpy.random.seed(123456789)
    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)
    swift_config_keys = args.swift_config.split(",")
    # check that all the keys are valid
    for c in swift_config_keys:
        try:
            SWIFT_CONFIGS[c]
        except KeyError as error:
            raise KeyError(
                f"Provided argument ({c}) does not match any swift "
                f"configuration keys. Please consult src/swift_configs.py "
                f"for available options."
            ) from error
    dask_client = set_up_dask(scheduler_address=scheduler)
    for config_key in swift_config_keys:
        log.info("Running for swift-config: %s", config_key)
        with performance_report(filename=f"dask-report-{config_key}.html"):
            ms_df = run_facet_to_subgrid(SWIFT_CONFIGS[config_key])
            ms_df.to_csv(f"seq_more_seq_{config_key}.csv")
        dask_client.restart()
    tear_down_dask(dask_client)


if __name__ == "__main__":
    #   "8k[1]-n4k-512",  1.546875
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
