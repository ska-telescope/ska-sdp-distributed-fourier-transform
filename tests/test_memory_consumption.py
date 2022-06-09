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
from distributed.diagnostics import MemorySampler

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import make_subgrid_and_facet
from src.swift_configs import SWIFT_CONFIGS
from src.utils import generate_input_data

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def sum_and_finish_subgrid(
    distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBFs
):
    """
    Combined function with Sum and Generate Subgrid

    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param facet_ixs: facet index list
    :param i0: i0 index
    :param i1: i1 index
    :param facet_ixs: facet index list
    :param NMBF_NMBFs: list of NMBF_NMBF graph

    :return: i0,i1 index, the shape of approx_subgrid
    """
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distr_fft.xM_size, distr_fft.xM_size), dtype=complex
    )
    # Add contributions
    for (j0, j1), NMBF_NMBF in zip(facet_ixs, NMBF_NMBFs):
        summed_facet += distr_fft.add_facet_contribution(
            distr_fft.add_facet_contribution(
                NMBF_NMBF,
                distr_fft.facet_off[j0],
                axis=0,
                use_dask=False,
                nout=1,
            ),
            distr_fft.facet_off[j1],
            axis=1,
            use_dask=False,
        )

    # Finish
    approx_subgrid = distr_fft.finish_subgrid(
        summed_facet,
        base_arrays.subgrid_A[i0],
        base_arrays.subgrid_A[i1],
        use_dask=False,
        nout=1,
    )

    return i0, i1, approx_subgrid.shape


def wait_for_tasks(work_tasks, timeout=None, return_when="ALL_COMPLETED"):
    """
    Simple function for waiting for tasks to finish.
    Logs completed tasks, and returns list of still-waiting tasks.

    :param work_tasks: task list
    :param timeout: timeout for waiting a task finshed
    :param return_when: return string

    :return: unfinshed task
    """

    # Wait for any task to finish
    dask.distributed.wait(work_tasks, timeout, return_when)

    # Remove finished tasks from work queue, return
    new_work_tasks = []
    for task in work_tasks:
        if task.done():
            log.info(f"Finished:{task.result()}")
        elif task.cancelled():
            log.info("Cancelled?")
        else:
            new_work_tasks.append(task)
    return new_work_tasks


def run_distributed_fft(
    fundamental_params, base_arrays, client, use_dask=True
):
    """
    A variation of the execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    Do not use HDF5 for simplification, and always uses Dask.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param client: Dask client

    :return: ms_df: memory information
    """

    distr_fft = StreamingDistributedFFT(**fundamental_params)

    # Generate the simplest input data
    G_2, FG_2 = generate_input_data(distr_fft, add_sources=False)

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
    N = distr_fft.N
    yB_size = distr_fft.yB_size
    yN_size = distr_fft.yN_size
    yP_size = distr_fft.yP_size
    xM_size = distr_fft.xM_size
    xM_yN_size = xM_size * yN_size // N

    log.info(" == Expected memory usage:")
    nfacet2 = distr_fft.nfacet**2
    max_work_columns = (
        1 + (max_work_tasks + distr_fft.nsubgrid - 1) // distr_fft.nsubgrid
    )
    BF_F_size = cpx_size * nfacet2 * yB_size * yP_size
    NMBF_BF_size = max_work_columns * cpx_size * nfacet2 * yP_size * xM_yN_size
    NMBF_NMBF_size = (
        max_work_tasks * cpx_size * nfacet2 * xM_yN_size * xM_yN_size
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
                        0,
                        base_arrays.Fb,
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
                                0,
                                distr_fft.subgrid_off[i0],
                                base_arrays.facet_m0_trunc,
                                base_arrays.Fn,
                                use_dask=use_dask,
                                nout=1,
                            ),
                            1,
                            base_arrays.Fb,
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
                            1,
                            distr_fft.subgrid_off[i1],
                            base_arrays.facet_m0_trunc,
                            base_arrays.Fn,
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


# @pytest.mark.parametrize(
#     "test_config, expected_result",
#     [("8k[1]-n4k-512", 1.546875), ("4k[1]-n2k-512", 4.1524e-1)],
# )
def memory_consumption(test_config, expected_result, save_data=False):
    """
    Main function to run the Distributed FFT
    For pipeline it does not save the data.
    If you'd like to examine the data independently, set save_data=True

    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    base_arrays = BaseArrays(**SWIFT_CONFIGS[test_config])
    dask_client = set_up_dask(scheduler_address=scheduler)

    log.info("Dask client setup %s", dask_client)
    log.info("Running for swift-config: %s", test_config)
    ms_df = run_distributed_fft(
        SWIFT_CONFIGS[test_config], base_arrays, client=dask_client
    )
    if save_data:
        ms_df.to_csv(f"ms_{test_config}.csv")

    # turn pandas DataFrame into numpy array
    data_array = ms_df["NMBF_NMBF.managed"].to_numpy()
    data_array = data_array / 1.0e9

    last_mem = data_array[-1]

    # BF_F size should have 16 bytes * nfacet * nfacet * yP_size * yB_size
    # Note: should assert the larst value used
    assert numpy.abs(last_mem - expected_result) / expected_result < 3
    tear_down_dask(dask_client)
