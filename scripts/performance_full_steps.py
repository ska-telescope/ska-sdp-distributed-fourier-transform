# pylint: disable=chained-comparison,too-many-arguments
# pylint: disable=unused-argument,too-many-statements
# pylint: disable=logging-fstring-interpolation，too-many-branches

"""
Testing Distributed Fourier Transform Testing for evaluating performance.
The 1st version was implemented by PM
"""

import argparse
import itertools
import logging
import os
import sys
import time

import dask
import dask.array
import dask.distributed
import h5py
import numpy
from distributed import performance_report
from distributed.diagnostics import MemorySampler
from matplotlib import pylab

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    extract_mid,
    make_facet_from_sources,
    make_subgrid_and_facet_from_hdf5,
    make_subgrid_from_sources,
)
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS
from src.utils import generate_input_data

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def sum_and_finish_subgrid(
    distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBFs
):
    """
    Sum all contributions and Finish subgrid

    :param distr_fft:
    :param base_arrays:
    :param i0:
    :param i1:
    :param facet_ixs:
    :param NMBF_NMBFs:

    :return:
    """
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distr_fft.xM_size, distr_fft.xM_size), dtype=complex
    )

    # Add contributions
    for (j0, j1), NMBF_NMBF in zip(facet_ixs, NMBF_NMBFs):
        summed_facet += distr_fft.add_facet_contribution(
            distr_fft.add_facet_contribution(
                NMBF_NMBF, distr_fft.facet_off[j0], axis=0
            ),
            distr_fft.facet_off[j1],
            axis=1,
        )

    # Finish
    return distr_fft.finish_subgrid(
        summed_facet, [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]]
    )


def prepare_and_split_subgrid(distr_fft, Fn, i0, i1, facet_ixs, subgrid):
    """

    :param distr_fft:
    :param Fn:
    :param i0:
    :param i1:
    :param facet_ixs:
    :param subgrid:
    :return:
    """
    # Prepare subgrid
    prepared_subgrid = distr_fft.prepare_subgrid(subgrid)

    # Extract subgrid facet contributions
    NAF_AFs = {
        j0: distr_fft.extract_subgrid_contrib_to_facet(
            prepared_subgrid, distr_fft.facet_off[j0], Fn, axis=0
        )
        for j0 in set(j0 for j0, j1 in facet_ixs)
    }
    NAF_NAFs = [
        distr_fft.extract_subgrid_contrib_to_facet(
            NAF_AFs[j0], distr_fft.facet_off[j1], Fn, axis=1
        )
        for j0, j1 in facet_ixs
    ]
    return NAF_NAFs


def make_facet(
    N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
):
    """

    :param N:
    :param yB_size:
    :param facet_off0:
    :param facet_B0:
    :param facet_off1:
    :param facet_B1:
    :param sources:
    :return:
    """
    # Create facet
    return make_facet_from_sources(
        sources, N, yB_size, [facet_off0, facet_off1], [facet_B0, facet_B1]
    )


def check_subgrid(N, sg_off0, sg_A0, sg_off1, sg_A1, approx_subgrid, sources):
    """

    :param N:
    :param sg_off0:
    :param sg_A0:
    :param sg_off1:
    :param sg_A1:
    :param approx_subgrid:
    :param sources:
    :return:
    """
    # Compare against subgrid (normalised)
    subgrid = make_subgrid_from_sources(
        sources, N, approx_subgrid.shape[0], [sg_off0, sg_off1], [sg_A0, sg_A1]
    )
    return numpy.sqrt(numpy.average(numpy.abs(subgrid - approx_subgrid) ** 2))


def check_facet(
    N, facet_off0, facet_B0, facet_off1, facet_B1, approx_facet, sources
):
    """

    :param N:
    :param facet_off0:
    :param facet_B0:
    :param facet_off1:
    :param facet_B1:
    :param approx_facet:
    :param sources:
    :return:
    """
    # Re-generate facet to compare against
    yB_size = approx_facet.shape[0]
    facet = make_facet(
        N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
    )

    # Compare against result
    return numpy.sqrt(numpy.average(numpy.abs(facet - approx_facet) ** 2))


def wait_for_tasks(work_tasks, timeout=None, return_when="ALL_COMPLETED"):
    """
    Simple function for waiting for tasks to finish.

    Logs completed tasks, and returns list of still-waiting tasks.
    :param work_tasks:
    :param timeout:
    :param return_when:

    :return:
    """

    # Wait for any task to finish
    dask.distributed.wait(
        [task for _, task in work_tasks], timeout, return_when
    )

    # Remove finished tasks from work queue, return
    new_work_tasks = []
    for name, task in work_tasks:
        if task.done():
            # If there's "{}" in the name, we should retrieve the
            # result and include it in the mesage.
            if "{}" in name:
                print(name.format(task.result()))
            else:
                print(name)
        elif task.cancelled():
            print("Cancelled", name)
        else:
            new_work_tasks.append((name, task))
    return new_work_tasks


def run_distributed_fft(
    fundamental_params,
    client=None,
):
    """
    Main execution function that reads in the configuration,
    generates the source data, and runs the algorithm.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param client: Dask client or None

    :returns: memory profile
    """
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    # Calculate expected memory usage
    MAX_WORK_TASKS = 20
    cpx_size = numpy.dtype(complex).itemsize
    N = fundamental_params["N"]
    yB_size = fundamental_params["yB_size"]
    yN_size = fundamental_params["yN_size"]
    yP_size = fundamental_params["yP_size"]
    xM_size = fundamental_params["xM_size"]
    xM_yN_size = xM_size * yN_size // N

    print(" == Expected memory usage:")
    nfacet2 = distr_fft.nfacet**2
    MAX_WORK_COLUMNS = (
        1 + (MAX_WORK_TASKS + distr_fft.nsubgrid - 1) // distr_fft.nsubgrid
    )
    BF_F_size = 2 * cpx_size * nfacet2 * yB_size * yP_size
    NMBF_BF_size = (
        2 * MAX_WORK_COLUMNS * cpx_size * nfacet2 * yP_size * xM_yN_size
    )
    NMBF_NMBF_size = (
        2 * MAX_WORK_TASKS * cpx_size * nfacet2 * xM_yN_size * xM_yN_size
    )
    print(f"BF_F (facets):                     {BF_F_size/1e9:.05} GB")
    print(f"NMBF_BF (subgrid columns):         {NMBF_BF_size/1e9:.05} GB")
    print(f"NMBF_NMBF (subgrid contributions): {NMBF_NMBF_size/1e9:.05} GB")
    print(
        f"Sum:                               {(BF_F_size+NMBF_BF_size+NMBF_NMBF_size)/1e9:.05} GB"
    )

    # Make facets containing just one source
    sources = [(1, 1, 0)]
    facet_2 = [
        [
            dask.delayed(make_facet)(
                N,
                yB_size,
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
    print("Facet offsets: ", distr_fft.facet_off)
    print("Subgrid offsets: ", distr_fft.subgrid_off)

    # List of all facet indices
    facet_ixs = list(
        itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet))
    )

    # Broadcast work arrays to all clients so they don't get re-send to
    # clients with every call.
    base_arrays_task = client.scatter(base_arrays, broadcast=True)
    Fb_task = client.scatter(base_arrays.Fb, broadcast=True)
    m_task = client.scatter(base_arrays.facet_m0_trunc, broadcast=True)
    Fn_task = client.scatter(base_arrays.Fn, broadcast=True)

    ms = MemorySampler()
    with ms.sample("process", measure="process"):
        with ms.sample("managed", measure="managed"):

            # ** Step 1
            # - forward: Facet preparation first axis
            # - backward: Gather inputs for facet finishing first axis

            # Generate BF_F terms (parallel over all facets)
            BF_F_tasks = dask.persist(
                [
                    distr_fft.prepare_facet(
                        facet_2[j0][j1],
                        Fb_task,
                        axis=0,
                        use_dask=True,
                        nout=1,
                    )
                    for j0, j1 in facet_ixs
                ]
            )[0]

            # Make tasks for accumulating facet data (equivalent of BF_F
            # for backwards direction). Generating them from BF_F_tasks is
            # a small cahet to encourage Dask to put the facets on the
            # same node. Not sure it is a good idea - alternatively just go
            # numpy.zeros((xP_size, yB_size), dtype=complex) here.
            MNAF_BMNAF_tasks = dask.persist(
                [
                    dask.delayed(lambda BF_F: numpy.zeros_like(BF_F))(
                        BF_F_task
                    )
                    for BF_F_task in BF_F_tasks
                ]
            )[0]

            # Job queue
            work_tasks = []

            # Sequentially go over subgrid columns
            for i0 in list(range(distr_fft.nsubgrid)):

                # ** Step 2
                # - forward: Extraction first axis, preparation second axis
                # - backward: Gather inputs for finishing second axis

                # Generate NMBF_BF terms (parallel over all facets)
                def extract_column(subgrid_off0, BF_F, Fn, m, Fb):
                    NMBF_F = distr_fft.extract_facet_contrib_to_subgrid(
                        BF_F, subgrid_off0, m, Fn, axis=0
                    )
                    return distr_fft.prepare_facet(NMBF_F, Fb, axis=1)

                NMBF_BF_tasks = dask.persist(
                    [
                        dask.delayed(extract_column)(
                            distr_fft.subgrid_off[i0],
                            BF_F_task,
                            Fn_task,
                            m_task,
                            Fb_task,
                        )
                        for BF_F_task in BF_F_tasks
                    ]
                )[0]

                # Make tasks for accumulating column data. Same slight
                # hack as mentioned above.
                NAF_MNAF_tasks = dask.persist(
                    [
                        dask.delayed(
                            lambda NMBF_BF: numpy.zeros_like(NMBF_BF)
                        )(NMBF_BF_task)
                        for NMBF_BF_task in NMBF_BF_tasks
                    ]
                )[0]

                # Sequential go over individual subgrids
                sleep_tasks = []
                for i1 in list(range(distr_fft.nsubgrid)):

                    # No space in work queue?
                    while len(work_tasks) >= MAX_WORK_TASKS:
                        work_tasks = wait_for_tasks(
                            work_tasks, return_when="FIRST_COMPLETED"
                        )

                    # ** Step 3 forward: Extraction second axis

                    # Generate NMBF_NMBF tasks (parallel over all facets)
                    NMBF_NMBF_tasks = [
                        distr_fft.extract_facet_contrib_to_subgrid(
                            NMBF_BF,
                            distr_fft.subgrid_off[i1],
                            m_task,
                            Fn_task,
                            axis=1,
                            use_dask=True,
                            nout=1,
                        )
                        for (j0, j1), NMBF_BF in zip(facet_ixs, NMBF_BF_tasks)
                    ]

                    # ** Step 4
                    # - forward: Sum and finish subgrids
                    # As a single Dask task - no point in splitting this
                    subgrid_task = dask.delayed(sum_and_finish_subgrid)(
                        distr_fft,
                        base_arrays_task,
                        i0,
                        i1,
                        facet_ixs,
                        NMBF_NMBF_tasks,
                    )

                    # Check erorr
                    check_task = dask.compute(
                        dask.delayed(check_subgrid)(
                            distr_fft.N,
                            distr_fft.subgrid_off[i0],
                            base_arrays.subgrid_A[i0],
                            distr_fft.subgrid_off[i1],
                            base_arrays.subgrid_A[i1],
                            subgrid_task,
                            sources,
                        ),
                        sync=False,
                    )[0]
                    work_tasks.append(
                        (f"Subgrid {i0}/{i1} error: {{}}", check_task)
                    )
                    del check_task

                    # - backward: Prepare and split subgrid
                    # Use nout to declare that we want to use
                    # contributions to different facets separately from
                    # here on out.
                    NAF_NAF_tasks = dask.delayed(
                        prepare_and_split_subgrid, nout=len(facet_ixs)
                    )(distr_fft, Fn_task, i0, i1, facet_ixs, subgrid_task)

                    # ** Step 3 backward: Accumulate on column
                    def accumulate_column(NAF_NAF, NAF_MNAF, m):
                        # TODO: add_subgrid_contribution should add
                        # directly to NAF_MNAF here at some point.
                        return NAF_MNAF + distr_fft.add_subgrid_contribution(
                            NAF_NAF,
                            distr_fft.subgrid_off[i1],
                            m,
                            axis=1,
                        )

                    NAF_MNAF_tasks = dask.compute(
                        [
                            dask.delayed(accumulate_column)(
                                NAF_NAF_task, NAF_MNAF_task, m_task
                            )
                            for NAF_NAF_task, NAF_MNAF_task in zip(
                                NAF_NAF_tasks, NAF_MNAF_tasks
                            )
                        ],
                        sync=False,
                    )[0]

                    # Add to work queue with an identifying name
                    work_tasks += [
                        (
                            f"Facet {j0}/{j1} subgrid {i0}/{i1} accumulated",
                            NAF_MNAF_task,
                        )
                        for (j0, j1), NAF_MNAF_task in zip(
                            facet_ixs, NAF_MNAF_tasks
                        )
                    ]

                # Step 2 backward: Accumulate facet
                def accumulate_facet(NAF_MNAF, MNAF_BMNAF, Fb, m, j1):

                    NAF_BMNAF = distr_fft.finish_facet(
                        NAF_MNAF, base_arrays.facet_B[j1], Fb, axis=1
                    )

                    # TODO: add_subgrid_contribution should add
                    # directly to NAF_MNAF here at some point.
                    MNAF_BMNAF = (
                        MNAF_BMNAF
                        + distr_fft.add_subgrid_contribution(
                            NAF_BMNAF, distr_fft.subgrid_off[i0], m, axis=0
                        )
                    )
                    return MNAF_BMNAF

                MNAF_BMNAF_tasks = dask.compute(
                    [
                        dask.delayed(accumulate_facet)(
                            NAF_MNAF_task, MNAF_BMNAF_task, Fb_task, m_task, j1
                        )
                        for NAF_MNAF_task, MNAF_BMNAF_task, (j0, j1) in zip(
                            NAF_MNAF_tasks, MNAF_BMNAF_tasks, facet_ixs
                        )
                    ],
                    sync=False,
                )[0]

                # Explicitly drop the reference to NAF_MNAF_tasks to
                # signal that Dask can free it immediately once the above
                # task(s) are finished
                del NAF_MNAF_tasks

            # Finish facets
            approx_facet_tasks = [
                distr_fft.finish_facet(
                    MNAF_BMNAF_task,
                    base_arrays.facet_B[j0],
                    Fb_task,
                    axis=0,
                    use_dask=True,
                    nout=1,
                )
                for MNAF_BMNAF_task, (j0, j1) in zip(
                    MNAF_BMNAF_tasks, facet_ixs
                )
            ]
            # Again, explicitly drop reference to prevent accumulated
            # facet data to stay around for longer than required
            del MNAF_BMNAF_tasks

            # Get error
            wait_for_tasks(work_tasks)
            print(
                "Facet errors: ",
                dask.compute(
                    [
                        dask.delayed(check_facet)(
                            distr_fft.N,
                            distr_fft.facet_off[j0],
                            base_arrays.facet_B[j0],
                            distr_fft.facet_off[j1],
                            base_arrays.facet_B[j1],
                            approx_facet_task,
                            sources,
                        )
                        for (j0, j1), approx_facet_task in zip(
                            facet_ixs, approx_facet_tasks
                        )
                    ]
                ),
            )

    ms_df = ms.to_pandas()
    return ms_df


def main(args):
    """
    Main function to run the Distributed FFT
    """
    ttt = dask.config.get("distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_")
    print("MALLOC_TRIM_THRESHOLD_:", ttt)
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
        ms_df = run_distributed_fft(
            SWIFT_CONFIGS[config_key],
            client=dask_client,
        )
        ms_df.to_csv(f"ms_{config_key}.csv")
        dask_client.restart()

    # PW: Not sure why we would do this?
    # tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
