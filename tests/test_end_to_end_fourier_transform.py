"""
End-to-end tests.
Reference data: generated by src/fourier_transform/test_algorithm.py on 24 Jan 2022
Reference data are stored in: tests/test_data/reference_data/

Important: I commented the non-relevant code out in test_algorithm.py before running it.
i.e. when I needed the results for the 1D case, I only ran that; when I needed them for
the 2D case, I commented the 1D parts out. This is to make sure results don't change
due to numpy.random being called different number of times between reference code and
tested code.
"""
import glob
import logging
import os
from unittest.mock import patch, call
import numpy
from numpy.testing import assert_array_almost_equal
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask

# client, current_env_var = set_up_dask()
client = None
current_env_var = None

from src.fourier_transform_1d_dask import main as main_1d
from src.fourier_transform_2d_dask import main as main_2d

from tests.test_data.ref_data import (
    EXPECTED_NONZERO_SUBGRID_1D,
    EXPECTED_NONZERO_FACET_1D,
    EXPECTED_NONZERO_APPROX_SUBGRID_1D,
    EXPECTED_NONZERO_APPROX_FACET_1D,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.WARNING)


def _compare_images(expected, result):
    with open(expected, "rb") as f1, open(result, "rb") as f2:
        contents1 = f1.read()
        contents2 = f2.read()
        try:
            assert contents1 == contents2
        except AssertionError:
            log.error("Assertion Failed: %s, %s", expected, result)
            raise AssertionError


def test_end_to_end_1d_dask():
    """
    Reference/expected values generated with numpy.random.seed(123456789)
    """
    result_subgrid, result_facet, result_approx_subgrid, result_approx_facet = main_1d(
        to_plot=False
    )

    # check array shapes
    assert result_subgrid.shape == (6, 188)
    assert result_facet.shape == (4, 256)
    assert result_approx_subgrid.shape == result_subgrid.shape
    assert result_approx_facet.shape == result_facet.shape

    # check array values
    assert_array_almost_equal(
        result_subgrid[numpy.where(result_subgrid != 0)],
        EXPECTED_NONZERO_SUBGRID_1D,
        decimal=9,
    )
    assert_array_almost_equal(
        result_facet[numpy.where(result_facet != 0)].round(8),
        EXPECTED_NONZERO_FACET_1D,
        decimal=7,
    )
    assert_array_almost_equal(
        result_approx_subgrid[numpy.where(result_approx_subgrid != 0)],
        EXPECTED_NONZERO_APPROX_SUBGRID_1D,
        decimal=9,
    )
    assert_array_almost_equal(
        result_approx_facet[numpy.where(result_approx_facet != 0)].round(8),
        EXPECTED_NONZERO_APPROX_FACET_1D,
        decimal=7,
    )

    if client:
        tear_down_dask(client, current_env_var)


# TODO: This logging test does not work for dask.delayed anymore. Please clean up
def test_end_to_end_1d_dask_logging():
    """
    Test that the logged information matches the
    expected listed in test_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    expected_log_calls = [
        call("xN=0.0207031 xM=0.125 yN=160 xNyN=3.3125 xA=0.0917969"),
        call("xN_size=42.4 xM_yP_size=128, xMxN_yP_size=150, xM_yN_size=80"),
        call("3x3 facets for FoV of 0.75 (100.0% efficiency)"),
        call("6 subgrids, 4 facets needed to cover"),
        # (facet to subgrid)
        call("Facet data: %s %s", (4, 256), 1024),
        call("Redistributed data: %s %s", (6, 4, 80), 1920),
        call("Reconstructed subgrids: %s %s", (6, 188), 1128),
        call("RMSE: %s (image: %s)", 4.2565524670253074e-08, 5.836290700482092e-07),
        # (subgrid to facet)
        call("Subgrid data: %s %s", (6, 188), 1128),
        call("Intermediate data: %s %s", (6, 4, 80), 1920),
        call("Reconstructed facets: %s %s", (4, 256), 1024),
        call("RMSE: %s (image: %s)", 1.1820730237627836e-07, 1.8913168380204536e-06),
    ]

    with patch("logging.Logger.info") as mock_log:
        main_1d(to_plot=False)
        for log_call in expected_log_calls:
            assert log_call in mock_log.call_args_list

    if client:
        tear_down_dask(client, current_env_var)


# TODO: The plotting test does not work for dask.delayed
def test_end_to_end_1d_dask_plot():
    """
    Test that the plots generated by the tested function match
    the relevant expected plots in test_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    fig_prefix = "test_data/test1d"
    plot_map_1d = {
        "test_data/reference_data/plot_n.png": f"{fig_prefix}_n.png",
        "test_data/reference_data/plot_fn.png": f"{fig_prefix}_fn.png",
        "test_data/reference_data/plot_xm.png": f"{fig_prefix}_xm.png",
        "test_data/reference_data/plot_error_facet_to_subgrid_1d.png": f"{fig_prefix}_error_facet_to_subgrid_1d.png",
        "test_data/reference_data/plot_error_subgrid_to_facet_1d.png": f"{fig_prefix}_error_subgrid_to_facet_1d.png",
    }

    main_1d(to_plot=True, fig_name=fig_prefix)

    for expected, result in plot_map_1d.items():
        _compare_images(expected, result)

    test_data = glob.glob(f"{fig_prefix}*.png")
    for f in test_data:
        os.remove(f)

    if client:
        tear_down_dask(client, current_env_var)


def test_end_to_end_2d_dask():
    """
    Test that the logged information matches the
    expected listed in test_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    expected_log_calls = [
        call("xN=0.0207031 xM=0.125 yN=160 xNyN=3.3125 xA=0.0917969"),
        call("xN_size=42.4 xM_yP_size=128, xMxN_yP_size=150, xM_yN_size=80"),
        call("3x3 facets for FoV of 0.75 (100.0% efficiency)"),
        call("6 subgrids, 4 facets needed to cover"),
        call("%s x %s subgrids %s x %s facets", 6, 6, 4, 4),
        call("Mean grid absolute: %s", 0.2523814510844513),
        # facet to subgrid
        call("RMSE: %s (image: %s)", 3.6351180911901923e-08, 6.834022011437562e-06),
        call("RMSE: %s (image: %s)", 1.8993992558912768e-17, 3.5708706010756e-15),
        # subgrid to facet - not yet added to tested code
        # call("RMSE: %s (image: %s)", 1.9503554118423179e-07, 4.992909854316334e-05),
        # call("RMSE: %s (image: %s)", 3.1048924152453573e-13, 7.948524583028115e-11),
    ]

    with patch("logging.Logger.info") as mock_log:
        main_2d(to_plot=False)
        for log_call in expected_log_calls:
            assert log_call in mock_log.call_args_list

    if client:
        tear_down_dask(client, current_env_var)


def test_end_to_end_2d_dask_plot():
    """
    Test that the plots generated by the tested function match
    the relevant expected plots in test_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    fig_prefix = "test_data/test2d"
    plot_map_2d = {
        "test_data/reference_data/plot_n.png": f"{fig_prefix}_n.png",
        "test_data/reference_data/plot_fn.png": f"{fig_prefix}_fn.png",
        "test_data/reference_data/plot_xm.png": f"{fig_prefix}_xm.png",
        # facet to subgrid
        "test_data/reference_data/plot_error_mean_facet_to_subgrid_2d.png": f"{fig_prefix}_error_mean_facet_to_subgrid_2d.png",
        "test_data/reference_data/plot_error_mean_image_facet_to_subgrid_2d.png": f"{fig_prefix}_error_mean_image_facet_to_subgrid_2d.png",
        "test_data/reference_data/plot_test_accuracy_facet_to_subgrid_2d.png": f"{fig_prefix}_test_accuracy_facet_to_subgrid_2d.png",
        # subgrid to facet -- not yet added to tested code
        # "test_data/reference_data/plot_error_mean_subgrid_to_facet_2d.png":
        #     f"{fig_prefix}_error_mean_subgrid_to_facet_2d.png",
        # "test_data/reference_data/plot_error_mean_image_subgrid_to_facet_2d.png":
        #     f"{fig_prefix}_error_mean_image_subgrid_to_facet_2d.png",
        # "test_data/reference_data/plot_test_accuracy_subgrid_to_facet_2d.png":
        #     f"{fig_prefix}_test_accuracy_subgrid_to_facet_2d.png",
    }

    main_2d(to_plot=True, fig_name=fig_prefix)

    for expected, result in plot_map_2d.items():
        _compare_images(expected, result)

    test_data = glob.glob(f"{fig_prefix}*.png")
    for f in test_data:
        os.remove(f)

    if client:
        tear_down_dask(client, current_env_var)
