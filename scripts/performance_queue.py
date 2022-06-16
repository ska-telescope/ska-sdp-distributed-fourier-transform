# pylint: disable=chained-comparison,too-many-arguments
# pylint: disable=unused-argument,too-many-statements
# pylint: disable=logging-fstring-interpolation，too-many-branches
"""
Testing Distributed Fourier Transform Testing for evaluating performance.
"""
import itertools
import logging
import os

import dask
import dask.array
import dask.distributed
import h5py
import numpy
from distributed import performance_report
from distributed.diagnostics import MemorySampler

from scripts.utils import (
    batch_all_i1_NMBF_NMBF,
    batch_NMBF_NMBF_sum_finish_subgrid,
    human_readable_size,
    wait_for_tasks,
    write_approx_subgrid,
    write_network_transfer_info,
)
from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    make_subgrid_and_facet_from_hdf5,
)
from src.fourier_transform_dask import cli_parser
from src.swift_configs import SWIFT_CONFIGS
from src.utils import error_task_facet_to_subgrid_2d, generate_input_data

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def cli_parser_with_batch():
    """
    Add some arguments for batch
    """
    parser = cli_parser()

    parser.add_argument(
        "--max_work_tasks",
        type=int,
        default=18,
        help="Maximum number of tasks in the queue",
    )
    parser.add_argument(
        "--max_NMBF_BF_waiting_task",
        type=int,
        default=4,
        help="NMBF_BF Number of waiting tasks",
    )
    parser.add_argument(
        "--batch_i1_number_task",
        type=int,
        default=26,
        help="How many i1's as a batch, eg: in 96k-n48k-512, "
        "this number is 26",
    )
    parser.add_argument(
        "--check_results",
        type=str,
        default="False",
        help="if save and check results",
    )
    return parser


@dask.delayed
def check_batch_i1(
    i0, i1_list, approx_subgrid_i1_list, true_subgrid_i1_list, base_arrays
):
    """
    check a batch of approx_subgrid

    :param i0: i0 index
    :param i1_list: a batch of i1 index
    :param approx_subgrid_i1_list: a batch of approx_subgrid
    :param true_subgrid_i1_list: a batch of true_subgrid
    :param base_arrays: BaseArrays class object

    :return List of i0,i1 and error of approx_subgrid
    """
    i1_res = []
    for i1 in i1_list:
        err_res = error_task_facet_to_subgrid_2d(
            approx_subgrid_i1_list[i1],
            true_subgrid_i1_list[i1],
            base_arrays.nsubgrid,
            use_dask=False,
            nout=1,
        )
        i1_res.append((i0, i1, numpy.average(err_res)))
    return i1_res


def check_approx_subgrid(
    fundamental_params,
    hdf5_prefix=None,
    hdf5_chunksize_G=256,
    hdf5_chunksize_FG=256,
):
    """
    check all approx subgrid using dask

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param hdf5_prefix: hdf5 path prefix
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data
    """
    base_arrays = BaseArrays(**fundamental_params)
    G_2_file = f"{hdf5_prefix}/G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
    FG_2_file = f"{hdf5_prefix}/FG_{base_arrays.N}_{hdf5_chunksize_FG}.h5"

    approx_G_2_file = (
        f"{hdf5_prefix}/approx_G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
    )

    true_subgrid, _ = make_subgrid_and_facet_from_hdf5(
        G_2_file,
        FG_2_file,
        base_arrays,
        use_dask=True,
    )

    approx_subgrid, _ = make_subgrid_and_facet_from_hdf5(
        approx_G_2_file,
        FG_2_file,
        base_arrays,
        use_dask=True,
    )
    err_res_task = []
    for i0 in range(base_arrays.nsubgrid):
        while len(err_res_task) >= 3:
            err_res_task = wait_for_tasks(err_res_task)
        approx_subgrid_i1_list = []
        true_subgrid_i1_list = []
        for i1 in range(base_arrays.nsubgrid):
            approx_subgrid_i1_list.append(approx_subgrid[i0][i1])
            true_subgrid_i1_list.append(true_subgrid[i0][i1])
        i1_res = check_batch_i1(
            i0,
            list(range(base_arrays.nsubgrid)),
            approx_subgrid_i1_list,
            true_subgrid_i1_list,
            base_arrays,
        )
        err_res_task += dask.compute(i1_res, sync=False)
    wait_for_tasks(err_res_task)


def run_facet_to_subgrid_with_batch(
    fundamental_params,
    use_dask=True,
    hdf5_prefix=None,
    hdf5_chunksize_G=256,
    hdf5_chunksize_FG=256,
    max_work_tasks=9,
    max_NMBF_BF_waiting_task=3,
    batch_i1_number_task=26,
    check_results=False,
):
    """
    Main execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param to_plot: run plotting?
    :param fig_name: If given, figures are saved with this prefix into
                     PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    :param use_dask: boolean; use Dask?
    :param hdf5_prefix: hdf5 path prefix
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data
    :return: dataframe of MemorySampler
    """
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)
    G_2_file = f"{hdf5_prefix}/G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
    FG_2_file = f"{hdf5_prefix}/FG_{base_arrays.N}_{hdf5_chunksize_FG}.h5"

    if (not os.path.exists(G_2_file)) and (not os.path.exists(FG_2_file)):
        G_2, FG_2 = generate_input_data(distr_fft)

    if not os.path.exists(G_2_file):
        with h5py.File(G_2_file, "w") as f:
            G_dataset = f.create_dataset(
                "G_data",
                G_2.shape,
                dtype="complex128",
                chunks=(hdf5_chunksize_G, hdf5_chunksize_G),
            )
            G_dataset[:] = G_2[:]

    if not os.path.exists(FG_2_file):
        with h5py.File(FG_2_file, "w") as f:
            FG_dataset = f.create_dataset(
                "FG_data",
                FG_2.shape,
                dtype="complex128",
                chunks=(hdf5_chunksize_FG, hdf5_chunksize_FG),
            )
            FG_dataset[:] = FG_2.astype("complex128")[:]

    if check_results:
        # create empty approx_G2 hdf5
        approx_G_2_file = (
            f"{hdf5_prefix}/approx_G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
        )
        if not os.path.exists(approx_G_2_file):
            with h5py.File(approx_G_2_file, "w") as f:
                f.create_dataset(
                    "G_data",
                    (base_arrays.N, base_arrays.N),
                    dtype="complex128",
                    chunks=(hdf5_chunksize_G, hdf5_chunksize_G),
                )

    _, facet_2 = make_subgrid_and_facet_from_hdf5(
        G_2_file,
        FG_2_file,
        base_arrays,
        use_dask=use_dask,
    )
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

            all_i0_list = list(range(distr_fft.nsubgrid))
            batch_i0_list = []
            for i in range(0, len(all_i0_list), max_NMBF_BF_waiting_task):
                batch_i0_list.append(
                    all_i0_list[i : i + max_NMBF_BF_waiting_task]
                )

            for batch_i0 in batch_i0_list:
                NMBF_BF_tasks_batch = []
                for i0 in batch_i0:
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
                    NMBF_BF_tasks_batch.append(NMBF_BF_tasks)

                for idx_i0, i0 in enumerate(batch_i0):
                    # ** Step 3: Extraction second axis
                    # Sequential go over individual subgrids

                    while len(work_tasks) >= max_work_tasks:
                        work_tasks = wait_for_tasks(
                            work_tasks, return_when="FIRST_COMPLETED"
                        )

                    all_i1_list = list(range(distr_fft.nsubgrid))
                    batch_i1_list = []
                    for i in range(0, len(all_i1_list), batch_i1_number_task):
                        batch_i1_list.append(
                            all_i1_list[i : i + batch_i1_number_task]
                        )

                    # compute NMBF_NMBF with all i1 axis in one task
                    batch_all_i1_NMBF_NMBF_list = []
                    for (j0, j1), NMBF_BF in zip(
                        facet_ixs, NMBF_BF_tasks_batch[idx_i0]
                    ):
                        batch_all_i1_NMBF_NMBF_list.append(
                            dask.delayed(
                                batch_all_i1_NMBF_NMBF, nout=len(batch_i1_list)
                            )(NMBF_BF, distr_fft, base_arrays, batch_i1_list)
                        )

                    for idx_i1, i1_batch in enumerate(batch_i1_list):

                        nfacet_ni1_batch_list = []
                        for F_batch in batch_all_i1_NMBF_NMBF_list:
                            nfacet_ni1_batch_list.append(F_batch[idx_i1])

                        batch_subgrid_task = (
                            batch_NMBF_NMBF_sum_finish_subgrid(
                                nfacet_ni1_batch_list,
                                distr_fft,
                                base_arrays,
                                facet_ixs,
                                i0,
                                i1_batch,
                                check=check_results,
                            )
                        )

                        if check_results:
                            batch_subgrid_task = write_approx_subgrid(
                                batch_subgrid_task,
                                base_arrays,
                                approx_G_2_file,
                            )
                        work_tasks += dask.compute(
                            batch_subgrid_task, sync=False
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
        # step: facet to approx_subgrid and write hdf5
        with performance_report(filename=f"dask-report-{config_key}.html"):
            ms_df = run_facet_to_subgrid_with_batch(
                SWIFT_CONFIGS[config_key],
                hdf5_prefix=args.hdf5_prefix,
                max_work_tasks=args.max_work_tasks,
                max_NMBF_BF_waiting_task=args.max_NMBF_BF_waiting_task,
                batch_i1_number_task=args.batch_i1_number_task,
                check_results=args.check_results == "True",
            )
            ms_df.to_csv(f"seq_more_seq_{config_key}.csv")

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
            "transfer_info.txt", f"{config_key},{tmp_size_1},{tmp_size_2}"
        )
        dask_client.compute(write_task, sync=True)

        # step: check subgrid and approx_subgrid
        if args.check_results == "True":
            check_approx_subgrid(
                SWIFT_CONFIGS[config_key],
                hdf5_prefix=args.hdf5_prefix,
            )

        dask_client.restart()
    tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser_with_batch()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
