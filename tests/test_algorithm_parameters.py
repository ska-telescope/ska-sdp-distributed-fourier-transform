"""
Unit tests for algorithm_parameters.py functions
"""

import numpy
import pytest

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    BaseParameters,
    StreamingDistributedFFT,
)
from tests.test_integration_fourier_transform import TEST_PARAMS


def test_base_params_fundamental():
    """
    Input dictionary values are correctly assigned to
    fundamental attributes of the class.
    """
    result = BaseParameters(**TEST_PARAMS)

    for k, v in TEST_PARAMS.items():
        assert result.__getattribute__(k) == v


def test_base_params_derived():
    """
    Input dictionary values are correctly used to
    obtain derived attributes of the class.
    """
    expected_derived = {
        "xM_yP_size": 128,
        "xM_yN_size": 80,
        "xMxN_yP_size": 150,
        "xN_yP_size": 22,
        "nsubgrid": 6,
        "nfacet": 4,
    }

    expected_facet_off = numpy.array([0, 256, 512, 768])
    expected_subgrid_off = numpy.array([0, 188, 376, 564, 752, 940])

    result = BaseParameters(**TEST_PARAMS)

    for k, v in expected_derived.items():
        assert result.__getattribute__(k) == v

    assert (result.facet_off == expected_facet_off).all()
    assert (result.subgrid_off == expected_subgrid_off).all()


def test_base_params_check_params():
    """
    BaseParameters.check_params is called as part of __init__
    It raises a ValueError if a certain condition doesn't apply,
    which can be achieved by slightly altering, e.g. N
    """
    new_params = TEST_PARAMS.copy()
    new_params["N"] = 1050

    with pytest.raises(ValueError):
        BaseParameters(**new_params)


def test_base_arrays_generate_mask():
    """
    Using subgrid_off and xA_size and nsubgrid, as would
    the code with the values specified by TEST_PARAMS
    """
    array_class = BaseArrays(**TEST_PARAMS)

    mask_size = 188
    offsets = [4, 192, 380, 568, 756, 944]

    # pylint: disable=protected-access
    mask = array_class._generate_mask(mask_size, offsets)
    assert mask.shape == (len(offsets), mask_size)
    assert (mask[0, :52] == 0.0).all()
    assert (mask[5, -52:] == 0.0).all()
    assert (mask[1:5, :] == 1.0).all()
    assert (mask[0, 53:] == 1.0).all()
    assert (mask[5, :-52] == 1.0).all()


def test_base_arrays_pure_arrays():
    """
    Fb, Fn, facet_m0_trunc, pswf are complicated arrays and their code is
    based on pure calculations, therefore I decided not to test
    their actual values, only that the code calculating them doesn't break
    when the class is instantiated with correct parameters.

    Note: F841 flake8 error ignored: "assigned but not used variable"
    """
    # pylint: disable=unused-variable
    array_class = BaseArrays(**TEST_PARAMS)
    fb = array_class.Fb  # noqa: F841
    fn = array_class.Fn  # noqa: F841
    facet_m0_trunc = array_class.facet_m0_trunc  # noqa: F841
    pswf = array_class.pswf  # noqa: F841


def test_basic_facet_to_subgrid():
    """Test basic properties of 1D facet to subgrid DFT for cases where
    the subgrids are expected to be a constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    Nx = TEST_PARAMS["Nx"] * 8  # allow more facet offsets
    Ny = N // Nx
    yB_size = TEST_PARAMS["yB_size"]
    assert yB_size % Ny == 0

    # Instantiate classes
    array_class = BaseArrays(**TEST_PARAMS)
    dft = StreamingDistributedFFT(**TEST_PARAMS)
    arr_pars = {
        "Fb": array_class.Fb,
        "Fn": array_class.Fn,
        "facet_m0_trunc": array_class.facet_m0_trunc,
    }

    # Test linearity with different values
    for val in [0, 1, 0.1]:

        # Check different facet offsets
        for facet_off in numpy.arange(-yB_size // 2 + Ny, yB_size // 2, Ny):

            # Set value at centre of image (might be off-centre for
            # the facet depending on offset)
            facet = numpy.zeros(yB_size)
            facet[yB_size // 2 - facet_off] = val
            prepped = dft.prepare_facet(facet, axis=0, **arr_pars)

            # Now generate subgrids at different (valid) subgrid offsets.
            for sg_off in numpy.arange(0, N, Nx):
                subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                    prepped, 0, sg_off, **arr_pars
                )
                subgrid_acc = dft.add_facet_contribution(
                    subgrid_contrib, facet_off, 0
                )
                subgrid = dft.finish_subgrid(subgrid_acc)

                # Now the entire subgrid should have (close to) a
                # constant value
                numpy.testing.assert_array_almost_equal(subgrid, val / N)


def test_basic_subgrid_to_facet():
    """Test basic properties of 1D subgrid to facet DFT for cases where a
    subgrid is set to a constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    Nx = TEST_PARAMS["Nx"] * 8  # allow more facet offsets
    Ny = N // Nx
    xA_size = TEST_PARAMS["xA_size"]
    xM_size = TEST_PARAMS["xM_size"]
    yB_size = TEST_PARAMS["yB_size"]
    assert yB_size % Ny == 0

    # Instantiate classes
    array_class = BaseArrays(**TEST_PARAMS)
    dft = StreamingDistributedFFT(**TEST_PARAMS)
    arr_pars = {
        "Fb": array_class.Fb,
        "Fn": array_class.Fn,
        "facet_m0_trunc": array_class.facet_m0_trunc,
    }

    # Test linearity with different values
    for val in [0, 1, 0.1]:

        # Start with subgrids at different (valid) subgrid offsets.
        for sg_off in numpy.arange(0, N, Nx):

            # Constant-value subgrid
            prepped = dft.prepare_subgrid(
                (val / xA_size) * numpy.ones(xA_size)
            )

            # Check different facet offsets
            for facet_off in numpy.arange(
                -yB_size // 2 + Ny, yB_size // 2, Ny
            ):
                extracted = dft.extract_subgrid_contrib_to_facet(
                    prepped, facet_off, axis=0, **arr_pars
                )
                accumulated = dft.add_subgrid_contribution(
                    extracted, sg_off, axis=0, **arr_pars
                )
                facet = dft.finish_facet(
                    accumulated, numpy.ones(xM_size), axis=0, **arr_pars
                )

                # Check that we have value at centre of image
                numpy.testing.assert_array_almost_equal(
                    facet[yB_size // 2 - facet_off], val
                )
