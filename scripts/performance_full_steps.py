# pylint: disable=chained-comparison,too-many-arguments，cell-var-from-loop
# pylint: disable=unused-argument,too-many-statements，unnecessary-lambda
# pylint: disable=logging-fstring-interpolation，too-many-branches

"""
Testing Distributed Fourier Transform Testing for evaluating performance.
The 1st version was implemented by PM
"""

import itertools
import logging
import os

import dask
import dask.array
import dask.distributed
import numpy
from distributed.diagnostics import MemorySampler

from scripts.utils import human_readable_size, write_network_transfer_info
from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask
from src.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


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
                log.info("%s", str(name.format(task.result())))
            else:
                log.info("%s", str(name))
        elif task.cancelled():
            log.info("Cancelled %s", str(name))
        else:
            new_work_tasks.append((name, task))
    return new_work_tasks


def sum_and_finish_subgrid(
    distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBFs
):
    """
    Combined function with Sum and Generate Subgrid

    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param i0: i0 index
    :param i1: i1 index
    :param facet_ixs: facet index list
    :param NMBF_NMBFs: list of NMBF_NMBF graph

    :return: i0,i1 index, the shape of approx_subgrid
    """

    summed_facet = numpy.zeros(
        (distr_fft.xM_size, distr_fft.xM_size), dtype=complex
    )

    for (j0, j1), NMBF_NMBF in zip(facet_ixs, NMBF_NMBFs):
        summed_facet += distr_fft.add_facet_contribution(
            distr_fft.add_facet_contribution(
                NMBF_NMBF, distr_fft.facet_off[j0], axis=0
            ),
            distr_fft.facet_off[j1],
            axis=1,
        )

    return distr_fft.finish_subgrid(
        summed_facet, [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]]
    )


def prepare_and_split_subgrid(distr_fft, Fn, i0, i1, facet_ixs, subgrid):
    """
    prepare and split subgrid facet contributions

    :param distr_fft: StreamingDistributedFFT class object
    :param Fn: Fourier transform of gridding function
    :param i0: i0 index
    :param i1: i1 index
    :param facet_ixs: facet index list
    :param subgrid: subgrid

    :return: NAF_NAFs
    """

    prepared_subgrid = distr_fft.prepare_subgrid(subgrid)

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
    Generates a facet from a source list

    This basically boils down to adding pixels on a grid, taking into account
    that coordinates might wrap around. Length of facet_offsets tuple decides
    how many dimensions the result has.

    :param N: image size
    :param yB_size: Desired size of facet
    :param facet_off0: Offset tuple of facet mid-point
    :param facet_B0: Mask expressions (optional)
    :param facet_off1: Offset tuple of facet mid-point
    :param facet_B1: Mask expressions (optional)
    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :returns: Numpy array with facet data
    """

    return make_facet_from_sources(
        sources, N, yB_size, [facet_off0, facet_off1], [facet_B0, facet_B1]
    )


def check_subgrid(N, sg_off0, sg_A0, sg_off1, sg_A1, approx_subgrid, sources):
    """
    Compare against subgrid (normalised)
    :param N: image size
    :param sg_off0: Offset tuple of subgrid mid-point
    :param sg_A0: Mask expressions (optional)
    :param sg_off1: Offset tuple of subgrid mid-point
    :param sg_A1: Mask expressions (optional)
    :param approx_subgrid: approx_subgrid
    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :returns: Numpy array with facet data
    """
    subgrid = make_subgrid_from_sources(
        sources, N, approx_subgrid.shape[0], [sg_off0, sg_off1], [sg_A0, sg_A1]
    )
    return numpy.sqrt(numpy.average(numpy.abs(subgrid - approx_subgrid) ** 2))


def check_facet(
    N, facet_off0, facet_B0, facet_off1, facet_B1, approx_facet, sources
):
    """
    Compare against subgrid (normalised)
    :param N: image size
    :param facet_off0: Offset tuple of facet mid-point
    :param facet_B0: Mask expressions (optional)
    :param facet_off1: Offset tuple of facet mid-point
    :param facet_B1: Mask expressions (optional)
    :param approx_facet: approx_facet
    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :returns: Numpy array with facet data
    """

    # Re-generate facet to compare against
    yB_size = approx_facet.shape[0]
    facet = make_facet(
        N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
    )

    # Compare against result
    return numpy.sqrt(numpy.average(numpy.abs(facet - approx_facet) ** 2))


def run_distributed_fft(
    fundamental_params,
    max_work_tasks=20,
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
    log.info("BF_F (facets): %.3f GB", BF_F_size / 1e9)
    log.info("NMBF_BF (subgrid columns):    %.3f     GB", NMBF_BF_size / 1e9)
    log.info(
        "NMBF_NMBF (subgrid contributions): %.3f GB", NMBF_NMBF_size / 1e9
    )
    log.info("Sum: %.3f GB", (BF_F_size + NMBF_BF_size + NMBF_NMBF_size) / 1e9)

    # Make facets containing just one source
    sources = [(1, 1, 0)]
    facet_2 = [
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
    log.info("Facet offsets: %s", str(distr_fft.facet_off))
    log.info("Subgrid offsets: %s", str(distr_fft.subgrid_off))

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
            # a small cheat to encourage Dask to put the facets on the
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
                for i1 in list(range(distr_fft.nsubgrid)):

                    # No space in work queue?
                    while len(work_tasks) >= max_work_tasks:
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
            check_res = dask.compute(
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
            )
            log.info("Facet errors: %s", str(check_res))

    ms_df = ms.to_pandas()
    return ms_df


def main(args):
    """
    Main function to run the Distributed FFT
    """
    ttt = dask.config.get("distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_")
    log.info("MALLOC_TRIM_THRESHOLD_: %d", ttt)
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
            args.queue_size,
            client=dask_client,
        )
        ms_df.to_csv(f"ms_{config_key}.csv")

        dict_outgoing = dask_client.run(
            lambda dask_worker: dask_worker.outgoing_transfer_log
        )
        dict_incoming = dask_client.run(
            lambda dask_worker: dask_worker.incoming_transfer_log
        )

        sum_getitem_incoming = 0.0
        for di_key in dict_incoming.keys():
            for di_key2 in dict_incoming[di_key]:
                if "getitem" in str(di_key2["keys"]):
                    sum_getitem_incoming += di_key2["total"]
        log.info(
            f"sum_getitem_incoming transfer bytes: {sum_getitem_incoming}"
        )
        sum_getitem_outgoing = 0.0
        for do_key in dict_outgoing.keys():
            for do_key2 in dict_outgoing[do_key]:
                if "getitem" in str(do_key2["keys"]):
                    sum_getitem_outgoing += do_key2["total"]
        log.info(
            f"sum_getitem_outgoing transfer bytes: {sum_getitem_outgoing}"
        )
        tmp_size_1 = human_readable_size(sum_getitem_incoming)
        tmp_size_2 = (human_readable_size(sum_getitem_outgoing),)
        write_task = write_network_transfer_info(
            "transfer_info_full_step.txt",
            f"{config_key},{tmp_size_1},{tmp_size_2}",
        )
        dask_client.compute(write_task, sync=True)

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
