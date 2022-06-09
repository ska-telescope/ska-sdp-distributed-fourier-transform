# pylint: disable=chained-comparison,too-many-arguments
# pylint: disable=unused-argument,too-many-statements
"""
Testing Distributed Fourier Transform Testing for evaluating performance.
"""
import itertools
import logging
import os
import sys

import dask
import dask.array
import dask.distributed
import h5py
import numpy
from distributed import performance_report
from distributed.diagnostics import MemorySampler

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
from src.utils import generate_input_data

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def sum_and_finish_subgrid(
    distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBFs
):
    """
    Sum all subgrid
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
    """
    # Wait for any task to finish
    dask.distributed.wait(work_tasks, timeout, return_when)
    # Remove finished tasks from work queue, return
    new_work_tasks = []
    for task in work_tasks:
        if task.done():
            print("Finished:", task.result())
        elif task.cancelled():
            print("Cancelled?")
        else:
            new_work_tasks.append(task)
    return new_work_tasks


@dask.delayed
def batch_NMBF_NMBF_sum_finish_subgrid(
    NMBF_BF_tasks, distr_fft, base_arrays, facet_ixs, i0, i1_batch
):
    """
    Barch Compute NMBF_NMBF and subgrid of i0's in a dask task

    :param NMBF_BF_tasks: NMBF_BF graph
    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param facet_ixs: facet index list
    :param i0: i0 index
    :param i1_batch: batch i0 index list

    :return: approx subgrid index list and shape
    """
    approx_subgrid_i0_list = []
    for i1 in i1_batch:
        NMBF_NMBF_list = [
            distr_fft.extract_facet_contrib_to_subgrid(
                NMBF_BF,
                1,
                distr_fft.subgrid_off[i1],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                use_dask=False,
                nout=1,
            )
            for (j0, j1), NMBF_BF in zip(facet_ixs, NMBF_BF_tasks)
        ]
        approx_subgrid_fake_res = sum_and_finish_subgrid(
            distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBF_list
        )
        approx_subgrid_i0_list.append(approx_subgrid_fake_res)
    return approx_subgrid_i0_list


def run_distributed_fft(
    fundamental_params,
    to_plot=True,
    fig_name=None,
    use_dask=False,
    client=None,
    use_hdf5=False,
    hdf5_prefix=None,
    hdf5_chunksize_G=256,
    hdf5_chunksize_FG=256,
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
    :param client: Dask client or None
    :param use_hdf5: use Hdf5?
    :param hdf5_prefix: hdf5 path prefix
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data
    :return: subgrid_2, facet_2, approx_subgrid, approx_facet
                when use_hdf5=False
             subgrid_2_file, facet_2_file, approx_subgrid_2_file,
                approx_facet_2_file when use_hdf5=True
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
    _, facet_2 = make_subgrid_and_facet_from_hdf5(
        G_2_file,
        FG_2_file,
        base_arrays,
        use_dask=use_dask,
    )
    # Calculate expected memory usage
    MAX_WORK_TASKS = 9
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
    BF_F_size = cpx_size * nfacet2 * yB_size * yP_size
    NMBF_BF_size = MAX_WORK_COLUMNS * cpx_size * nfacet2 * yP_size * xM_yN_size
    NMBF_NMBF_size = (
        MAX_WORK_TASKS * cpx_size * nfacet2 * xM_yN_size * xM_yN_size
    )
    print(f"BF_F (facets):                     {BF_F_size / 1e9:.03} GB")
    print(f"NMBF_BF (subgrid columns):         {NMBF_BF_size / 1e9:.03} GB")
    print(f"NMBF_NMBF (subgrid contributions): {NMBF_NMBF_size / 1e9:.03} GB")
    print(
        "Sum:                               "
        + f"{(BF_F_size + NMBF_BF_size + NMBF_NMBF_size) / 1e9:.02} GB"
    )
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
            MAX_ppp_task = 3
            batch_i0_list = []
            for i in range(0, len(all_i0_list), MAX_ppp_task):
                batch_i0_list.append(all_i0_list[i : i + MAX_ppp_task])
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
                    all_i1_list = list(range(distr_fft.nsubgrid))
                    seq_i1_task = 10
                    batch_i1_list = []
                    for i in range(0, len(all_i1_list), seq_i1_task):
                        batch_i1_list.append(all_i1_list[i : i + seq_i1_task])
                    for i1_batch in batch_i1_list:
                        while len(work_tasks) >= MAX_WORK_TASKS:
                            work_tasks = wait_for_tasks(
                                work_tasks, return_when="FIRST_COMPLETED"
                            )
                        batch_subgrid_task = (
                            batch_NMBF_NMBF_sum_finish_subgrid(
                                NMBF_BF_tasks_batch[idx_i0],
                                distr_fft,
                                base_arrays,
                                facet_ixs,
                                i0,
                                i1_batch,
                            )
                        )
                        work_tasks += dask.compute(
                            batch_subgrid_task, sync=False
                        )
                    wait_for_tasks(work_tasks)
                    # clean batch
                    NMBF_BF_tasks_batch[idx_i0] = None
                # Finish tasks
                wait_for_tasks(work_tasks)
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
        with performance_report(filename=f"dask-report-{config_key}.html"):
            ms_df = run_distributed_fft(
                SWIFT_CONFIGS[config_key],
                to_plot=False,
                use_dask=True,
                hdf5_prefix="../../../bigresult",
                client=dask_client,
            )
            ms_df.to_csv(f"seq_more_seq_{config_key}.csv")
        dask_client.restart()
    tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
