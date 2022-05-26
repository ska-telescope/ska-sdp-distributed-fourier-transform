"""
    Test script for memory consumption (ORC-1247)
    Created by Feng Wang
    Currently it only tests the prepare facet step

"""
import itertools
import logging
import os
import sys
import time

import dask
import numpy
import pytest
from distributed.diagnostics import MemorySampler
from dask.distributed import wait
from numpy.testing import assert_almost_equal

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


def sleep_and_cut(x):
    """
    Function to let Dask sleep
    """
    time.sleep(5)
    return x.shape


def run_distributed_fft(fundamental_params, client):
    """
    A variation of the execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    Do not use HDF5 for simplification, and always uses Dask.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param client: Dask client

    :return: ms_df: memory information
    """

    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    # Generate the simplest input data
    G_2, FG_2 = generate_input_data(distr_fft, add_sources=False)

    log.info("------------------------------------------")

    G_2 = client.scatter(G_2)
    FG_2 = client.scatter(FG_2)

    _, facet_2 = make_subgrid_and_facet(
        G_2,
        FG_2,
        base_arrays,
        dims=2,
        use_dask=True,
    )

    facet_2 = dask.persist(facet_2)[0]
    wait(facet_2)

    BF_F_list = []
    for j0, j1 in itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet)):
        BF_F = distr_fft.prepare_facet(
            facet_2[j0][j1],
            0,
            base_arrays.Fb,
            use_dask=True,
            nout=1,
        )
        BF_F_list.append(BF_F)

    ms = MemorySampler()
    with ms.sample("BF_F", measure="managed"):
        BF_F_list = dask.persist(BF_F_list)[0]
        sleep_task = [dask.delayed(sleep_and_cut)(bf) for bf in BF_F_list]
        sleep_task = dask.compute(sleep_task, sync=True)
    for i in sleep_task:
        log.info("%s \n", i)
    ms_df = ms.to_pandas()
    return ms_df


def calculate_expected_memory(fundamental_params, max_work_tasks=1):

    """
    Calculate the theoretical upper limit on memory usage

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param max_work_tasks: maximum number of work tasks
    :return BF_F_size, NMBF_BF_size, NMBF_NMBF_size:
    """

    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    cpx_size = numpy.dtype(complex).itemsize
    N = fundamental_params["N"]
    yB_size = fundamental_params["yB_size"]
    yN_size = fundamental_params["yN_size"]
    yP_size = fundamental_params["yP_size"]
    xM_size = fundamental_params["xM_size"]
    xM_yN_size = xM_size * yN_size // N

    nfacet2 = distr_fft.nfacet**2
    max_work_columns = (
        1 + (max_work_tasks + distr_fft.nsubgrid - 1) // distr_fft.nsubgrid
    )
    BF_F_size = cpx_size * nfacet2 * yB_size * yP_size / 1e9
    NMBF_BF_size = max_work_columns * cpx_size * nfacet2 * yP_size * xM_yN_size / 1e9
    NMBF_NMBF_size = max_work_tasks * cpx_size * nfacet2 * xM_yN_size * xM_yN_size / 1e9

    return BF_F_size, NMBF_BF_size, NMBF_NMBF_size


@pytest.mark.parametrize(
    "test_config, expected_result",
    [("8k[1]-n4k-512", 1.546875), ("4k[1]-n2k-512", 4.1524e-1)],
)
def test_memory_consumption(test_config, expected_result, save_data=False):
    """
    Main function to run the Distributed FFT
    For pipeline it does not save the data.
    If you'd like to examine the data independently, set save_data=True

    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    dask_client = set_up_dask(scheduler_address=scheduler)

    log.info("Dask client setup %s", dask_client)
    log.info("Running for swift-config: %s", test_config)
    ms_df = run_distributed_fft(SWIFT_CONFIGS[test_config], client=dask_client)
    if save_data:
        ms_df.to_csv(f"ms_{test_config}.csv")

    # turn pandas DataFrame into numpy array
    data_array = ms_df["BF_F"].to_numpy()
    data_array = data_array / 1.0e9
    max_mem = numpy.max(data_array)
    avg_mem = numpy.mean(data_array)
    log.info("%s, %s", max_mem, avg_mem)

    # BF_F size should have 16 bytes * nfacet * nfacet * yP_size * yB_size
    # We also calculate the upper limit of the memory
    max_memory_expected, _, _ = calculate_expected_memory(
        SWIFT_CONFIGS[test_config], max_work_tasks=8
    )

    assert max_mem < mam_memory_expected
    tear_down_dask(dask_client)
