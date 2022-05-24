"""
    Test script for memory consumption
    Created by Feng Wang
    Currently it only tests Step 1
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


def run_distributed_fft(fundamental_params, use_dask=False, client=None):
    """
    A variation of the execution function that reads in the configuration,
    generates the source data, and runs the algorithm.
    Do not use HDF5 for simplification.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param use_dask: boolean; use Dask?
    :param client: Dask client or None

    :return: subgrid_2, facet_2, approx_subgrid, approx_facet
    """

    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    # Generate the simplest input data
    G_2, FG_2 = generate_input_data(distr_fft, add_sources=False)

    log.info("------------------------------------------")

    if use_dask and client is not None:
        G_2 = client.scatter(G_2)
        FG_2 = client.scatter(FG_2)

    _, facet_2 = make_subgrid_and_facet(
        G_2,
        FG_2,
        base_arrays,
        dims=2,
        use_dask=use_dask,
    )

    BF_F_list = []
    for j0, j1 in itertools.product(range(distr_fft.nfacet), range(distr_fft.nfacet)):
        BF_F = distr_fft.prepare_facet(
            facet_2[j0][j1],
            0,
            base_arrays.Fb,
            use_dask=use_dask,
            nout=1,
        )
        BF_F_list.append(BF_F)

    ms = MemorySampler()
    with ms.sample("BF_F"):
        BF_F_list = dask.persist(BF_F_list)[0]
        sleep_task = [dask.delayed(sleep_and_cut)(bf) for bf in BF_F_list]
        sleep_task = dask.compute(sleep_task, sync=True)
    for i in sleep_task:
        print(i)
    ms_df = ms.to_pandas()
    return ms_df


@pytest.mark.parametrize(
    "test_config, expected_result",
    [("1k[1]-n512-512", 2.5952e-2), ("4k[1]-n2k-512", 4.1524e-1)],
)
def test_memory_consumption(test_config, expected_result, save_data=False):
    """
    Main function to run the Distributed FFT
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    # check that all the keys are valid
    try:
        SWIFT_CONFIGS[test_config]
    except KeyError:
        pytest.fail(
            f"Provided argument ({test_config}) does not match any swift "
            f"configuration keys. Please consult src/swift_configs.py "
            f"for available options."
        )

    dask_client = set_up_dask(scheduler_address=scheduler)

    log.info("Dask client setup %s", dask_client)
    log.info("Running for swift-config: %s", test_config)
    ms_df = run_distributed_fft(
        SWIFT_CONFIGS[test_config], use_dask=True, client=dask_client
    )
    if save_data:
        ms_df.to_csv(f"ms_{test_config}.csv")

    # turn pandas DataFrame into numpy array
    data_array = ms_df["BF_F"].to_numpy()
    data_array = data_array / 1.0e9
    max_mem = numpy.max(data_array)
    avg_mem = numpy.mean(data_array)
    last_mem = data_array[-1]
    log.info("%s, %s", max_mem, avg_mem)

    # BF_F size should have 16 bytes * nfacet * nfacet * yP_size * yB_size
    # Note: should assert the last value used
    assert_almost_equal(last_mem, expected_result)
    tear_down_dask(dask_client)
