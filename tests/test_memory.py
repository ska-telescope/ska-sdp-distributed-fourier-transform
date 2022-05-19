"""
    Test script for memory consumption
    Created by Feng Wang
"""
import itertools
import logging
import os
import sys
import time
import h5py

import dask
import dask.array
import numpy
from distributed.diagnostics import MemorySampler

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    make_subgrid_and_facet_from_hdf5,
)
from src.swift_configs import SWIFT_CONFIGS
from src.utils import (
    generate_input_data,

)
from src.fourier_transform_dask import cli_parser

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

def sleep_and_cut(x):

    time.sleep(5)
    return x.shape

def run_distributed_fft(
    fundamental_params,
    use_dask=False,
    hdf5_prefix=None,
    hdf5_chunksize_G=256,
    hdf5_chunksize_FG=256,
):
    """
    A variation of the execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    For simplification, automatically assume use_hdf5=True.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param use_dask: boolean; use Dask?
    :param hdf5_prefix: hdf5 path prefix
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data

    :return: subgrid_2, facet_2, approx_subgrid, approx_facet
    """
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    G_2, FG_2 = generate_input_data(distr_fft)

    G_2_file = f"{hdf5_prefix}/G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
    FG_2_file = f"{hdf5_prefix}/FG_{base_arrays.N}_{hdf5_chunksize_FG}.h5"
    print(G_2_file)
    print(FG_2_file)

    print(G_2.shape)
    print(FG_2.shape)
    if not os.path.exists(G_2_file):
        with h5py.File(G_2_file, "w") as f:
            G_dataset = f.create_dataset(
                "G_data",
                G_2.shape,
                dtype="complex128",
                chunks=(hdf5_chunksize_G, hdf5_chunksize_G),
            )
            G_dataset[:] = G_2[:]
        
    print("---")
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

    BF_F_list = []
    for j0, j1 in itertools.product(
        range(distr_fft.nfacet), range(distr_fft.nfacet)
    ):
        BF_F = distr_fft.prepare_facet(
            facet_2[j0][j1],
            0,
            base_arrays.Fb,
            use_dask=use_dask,
            nout=1,
        )
        BF_F_list.append(BF_F)

    sleep_task = [dask.delayed(sleep_and_cut(bf)) for bf in BF_F_list]
    ms = MemorySampler()
    with ms.sample("BF_F"):
        sleep_task = dask.compute(sleep_task,sync=True)
    for i in sleep_task:
        print(sleep_task)
    ms_df = ms.to_pandas()
    return ms_df

        

def main(args):
    """
    Main function to run the Distributed FFT
    """
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
            use_dask=True,
            hdf5_prefix="../bigresult",
        )
        ms_df.to_csv(f"ms_{config_key}.csv")
        dask_client.restart()
    tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)