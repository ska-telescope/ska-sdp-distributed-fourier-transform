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
import logging
from unittest.mock import call, patch

import numpy
import pytest
from numpy.testing import assert_array_almost_equal

from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.algorithm_1d.fourier_transform_1d_dask import main as main_1d
from src.fourier_transform_2d_dask import main as main_2d
from tests.test_data.reference_data.ref_data import (
    EXPECTED_NONZERO_APPROX_FACET_1D,
    EXPECTED_NONZERO_APPROX_SUBGRID_1D,
    EXPECTED_NONZERO_FACET_1D,
    EXPECTED_NONZERO_SUBGRID_1D,
)

from tests.test_data.reference_data.ref_data_2d import (
    EXPECTED_NONZERO_SUBGRID_2D,
    EXPECTED_FACET_2D,
    EXPECTED_NONZERO_APPROX_FACET_2D,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.WARNING)

TARGET_PARS = {
    "W": 13.25,  # PSWF parameter (grid-space support)
    "fov": 0.75,  # field of view?
    "N": 1024,  # total image size
    "Nx": 4,  # subgrid spacing: it tells you what subgrid offsets are permissible:
    # here it is saying that they need to be divisible by 4.
    "yB_size": 256,  # true usable image size (facet)
    "yN_size": 320,  # padding needed to transfer the data?
    "yP_size": 512,  # padded (rough) image size (facet)
    "xA_size": 188,  # true usable subgrid size
    "xM_size": 256,  # padded (rough) subgrid size
}


@pytest.mark.parametrize(
    "use_dask, dask_option",
    [(False, None), (True, "delayed"), (True, "array")],
)
def test_end_to_end_1d_dask(use_dask, dask_option):
    """
    Test that the 1d algorithm produces the same results without dask,
    and with dask with array or delayed.
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    if use_dask:
        client = set_up_dask()

    (
        result_subgrid,
        result_facet,
        result_approx_subgrid,
        result_approx_facet,
    ) = main_1d(to_plot=False, use_dask=use_dask, dask_option=dask_option)

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

    if use_dask:
        tear_down_dask(client)


@pytest.mark.parametrize("use_dask", [False, True])
def test_end_to_end_2d_dask(use_dask):
    """
    Test that the 2d algorithm produces the same results with and without dask.
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    if use_dask:
        client = set_up_dask()

    (
        result_subgrid,
        result_facet,
        result_approx_subgrid,
        result_approx_facet,
    ) = main_2d(TARGET_PARS, to_plot=False, use_dask=use_dask)

    # check array shapes
    assert result_subgrid.shape == (6, 6, 188, 188)
    assert result_facet.shape == (4, 4, 256, 256)
    # TODO Why is this?
    # assert result_approx_subgrid.shape == (6, 6, 4, 4, 80, 80)
    assert result_approx_facet.shape == result_facet.shape

    # check array values
    result_subgrid_sliced = result_subgrid[:50, :50, :50, :50]
    result_subgrid_sliced_nonzero = result_subgrid_sliced[
        numpy.where(result_subgrid_sliced != 0)
    ]
    assert_array_almost_equal(
        result_subgrid_sliced_nonzero,
        EXPECTED_NONZERO_SUBGRID_2D,
        decimal=9,
    )

    assert_array_almost_equal(
        result_facet[numpy.where(result_facet != 0)].round(8),
        EXPECTED_FACET_2D,
        decimal=4,
    )

    result_approx_facet_sliced = result_approx_facet[:50, :50, :50, :50]
    result_approx_facet_sliced_nonzero = result_approx_facet_sliced[
        numpy.where(result_approx_facet_sliced != 0)
    ]
    assert_array_almost_equal(
        result_approx_facet_sliced_nonzero.round(8),
        EXPECTED_NONZERO_APPROX_FACET_2D,
        decimal=4,
    )

    if use_dask:
        tear_down_dask(client)


# this test does not seem to work with the gitlab-ci;
@pytest.mark.skip
@pytest.mark.parametrize("use_dask", [False, True])
def test_end_to_end_2d_dask_logging(use_dask):
    """
    Test that the logged information matches the
    expected listed in test_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    if use_dask:
        client = set_up_dask()

    # the values in this test slightly changed (10-5 - 10-10)
    # could this be because originally numpy.fft2 was used for the 2d version?
    expected_log_calls = [
        call("6 subgrids, 4 facets needed to cover"),
        call("%s x %s subgrids %s x %s facets", 6, 6, 4, 4),
        call("Mean grid absolute: %s", 0.25238145108445126),
        # facet to subgrid
        call(
            "RMSE: %s (image: %s)",
            3.635118091200949e-08,
            6.834022011457784e-06,
        ),
        call("RMSE: %s (image: %s)", 1.8993993540584405e-17, 3.5708707856298686e-15),
        # subgrid to facet - not yet added to tested code
        call(
            "RMSE: %s (image: %s)",
            1.906652955419094e-07,
            4.881031565872881e-05,
        ),
        call(
            "RMSE: %s (image: %s)",
            3.1048926297115777e-13,
            7.948525132061639e-11,
        ),
    ]

    with patch("logging.Logger.info") as mock_log:
        main_2d(TARGET_PARS, to_plot=False, use_dask=use_dask)
        for log_call in expected_log_calls:
            assert log_call in mock_log.call_args_list

    if use_dask:
        tear_down_dask(client)
