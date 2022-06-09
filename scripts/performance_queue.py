# pylint: disable=chained-comparison,too-many-arguments
# pylint: disable=unused-argument,too-many-statements
# pylint: disable=logging-fstring-interpolation
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

from scripts.utils import batch_NMBF_NMBF_sum_finish_subgrid, wait_for_tasks,  sum_and_finish_subgrid

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

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
                        while len(work_tasks) >= max_work_tasks:
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
