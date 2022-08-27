"""
Unit tests for algorithm_parameters.py functions
"""

import itertools

import numpy
import pytest

from ska_sdp_exec_swiftly.fourier_transform.algorithm_parameters import (
    BaseArrays,
    BaseParameters,
    StreamingDistributedFFT,
)
from ska_sdp_exec_swiftly.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)

from .test_integration_fourier_transform import TEST_PARAMS


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


def test_facet_to_subgrid_basic():
    """Test basic properties of 1D facet to subgrid distributed FT
    primitives for cases where the subgrids are expected to be a
    constant value.
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

    # Test with different values and facet offsets
    for val, facet_off in itertools.product(
        [0, 1, 0.1], numpy.arange(-yB_size // 2 + Ny, yB_size // 2, Ny)
    ):

        # Set value at centre of image (might be off-centre for
        # the facet depending on offset)
        facet = numpy.zeros(yB_size)
        facet[yB_size // 2 - facet_off] = val
        prepped = dft.prepare_facet(facet, axis=0, **arr_pars)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in numpy.arange(0, N, Nx):
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_off, axis=0, **arr_pars
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_contrib, facet_off, axis=0
            )
            subgrid = dft.finish_subgrid(subgrid_acc)

            # Now the entire subgrid should have (close to) a
            # constant value
            numpy.testing.assert_array_almost_equal(subgrid, val / N)


def test_facet_to_subgrid_dft_1d():
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    Nx = TEST_PARAMS["Nx"] * 8  # allow more facet offsets
    Ny = N // Nx
    xA_size = TEST_PARAMS["xA_size"]
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

    # Test with different values and facet offsets
    for sources, facet_off in itertools.product(
        [
            [(1, 0)],
            [(2, 1)],
            [(1, -3)],
            [(-0.1, 5)],
            [(1, 20), (2, 5), (3, -4)],
            [(0, i) for i in range(-20, 20)],
        ],
        numpy.arange(-yB_size // 2 + Ny, yB_size // 2, Ny),
    ):

        # Set sources in facet
        facet = make_facet_from_sources(sources, N, yB_size, [facet_off])

        # We assume all sources are on the facet
        assert numpy.sum(facet) == sum(src[0] for src in sources)
        prepped = dft.prepare_facet(facet, axis=0, **arr_pars)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in [0, Nx, -Nx, N]:
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_off, axis=0, **arr_pars
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_contrib, facet_off, axis=0
            )
            subgrid = dft.finish_subgrid(subgrid_acc)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            numpy.testing.assert_array_almost_equal(subgrid, expected)


def test_facet_to_subgrid_dft_2d():
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation -- 2D version
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    Nx = TEST_PARAMS["Nx"] * 8  # allow more facet offsets
    Ny = N // Nx
    xA_size = TEST_PARAMS["xA_size"]
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

    # Test with different values and facet offsets
    for sources, facet_offs in itertools.product(
        [
            [(1, 0, 0)],
            [(1, 20, 4), (2, 2, 5), (3, -5, -4)],
        ],
        [[0, 0], [Ny, Ny], [-Ny, Ny], [0, -Ny]],
    ):

        # Set sources in facet
        facet = make_facet_from_sources(sources, N, yB_size, facet_offs)

        # We assume all sources are on the facet
        assert numpy.sum(facet) == sum(src[0] for src in sources)
        prepped0 = dft.prepare_facet(facet, axis=0, **arr_pars)
        prepped = dft.prepare_facet(prepped0, axis=1, **arr_pars)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_offs in [[0, 0], [0, Nx], [Nx, 0], [-Nx, -Nx]]:
            subgrid_contrib0 = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_offs[0], axis=0, **arr_pars
            )
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                subgrid_contrib0, sg_offs[1], axis=1, **arr_pars
            )
            subgrid_acc0 = dft.add_facet_contribution(
                subgrid_contrib, facet_offs[0], axis=0
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_acc0, facet_offs[1], axis=1
            )
            subgrid = dft.finish_subgrid(subgrid_acc)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, sg_offs)
            numpy.testing.assert_array_almost_equal(subgrid, expected)


def test_subgrid_to_facet_basic():
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
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

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for val, sg_off in itertools.product([0, 1, 0.1], numpy.arange(0, N, Nx)):

        # Constant-value subgrid
        prepped = dft.prepare_subgrid((val / xA_size) * numpy.ones(xA_size))

        # Check different facet offsets
        for facet_off in numpy.arange(-yB_size // 2 + Ny, yB_size // 2, Ny):
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


def test_subgrid_to_facet_dft():
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
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

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for sources, sg_off in itertools.product(
        [
            [(1, 0)],
            [(2, 1)],
            [(1, -3)],
            [(-0.1, 5)],
        ],
        numpy.arange(0, N, Nx),
    ):

        # Generate subgrid. As we are only filling the grid partially
        # here, we have to scale it.
        subgrid = (
            make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            / xA_size
            * N
        )
        prepped = dft.prepare_subgrid(subgrid)

        # Check different facet offsets
        for facet_off in numpy.arange(-yB_size // 2 + Ny, yB_size // 2, Ny):
            extracted = dft.extract_subgrid_contrib_to_facet(
                prepped, facet_off, axis=0, **arr_pars
            )
            accumulated = dft.add_subgrid_contribution(
                extracted, sg_off, axis=0, **arr_pars
            )
            facet = dft.finish_facet(
                accumulated, numpy.ones(xM_size), axis=0, **arr_pars
            )

            # Check that pixels in questions have correct value. As -
            # again - we have only partially filled the grid, the only
            # value we can really check is the (singular) one we set
            # previously.
            expected = make_facet_from_sources(
                sources, N, yB_size, [facet_off]
            )
            numpy.testing.assert_array_almost_equal(
                facet[expected != 0], expected[expected != 0]
            )
