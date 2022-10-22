"""
Unit tests for algorithm_parameters.py functions
"""

import itertools

import numpy
import pytest

from ska_sdp_exec_swiftly.fourier_transform.algorithm_parameters import (
    SwiftlyCore,
)
from ska_sdp_exec_swiftly.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)
from ska_sdp_exec_swiftly.swift_configs import SWIFT_CONFIGS

TEST_PARAMS = {
    "W": 13.5625,
    "N": 1024,
    "yB_size": 416,
    "yN_size": 512,
    "xA_size": 228,
    "xM_size": 256,
}


def make_core(pars):
    """Construct SwiftlyCore from parameter dictionary"""
    return SwiftlyCore(pars["W"], pars["N"], pars["xM_size"], pars["yN_size"])


def test_base_params_fundamental():
    """
    Input dictionary values are correctly assigned to
    fundamental attributes of the class.
    """
    result = make_core(TEST_PARAMS)

    assert result.W == TEST_PARAMS["W"]
    assert result.N == TEST_PARAMS["N"]
    assert result.yN_size == TEST_PARAMS["yN_size"]
    assert result.xM_size == TEST_PARAMS["xM_size"]


def test_base_params_derived():
    """
    Input dictionary values are correctly used to
    obtain derived attributes of the class.
    """

    result = make_core(TEST_PARAMS)
    assert result.xM_yN_size == 128


def test_base_params_check_params():
    """
    BaseParameters.check_params is called as part of __init__
    It raises a ValueError if a certain condition doesn't apply,
    which can be achieved by slightly altering, e.g. N
    """
    new_params = TEST_PARAMS.copy()
    new_params["N"] = 1050

    with pytest.raises(ValueError):
        make_core(new_params)


def test_swift_configs():
    """
    Test all standard configurations
    """

    for config in SWIFT_CONFIGS.values():
        if config["N"] < 4 * 1024:
            make_core(config)
    """Test basic properties of 1D facet to subgrid distributed FT
    primitives for cases where the subgrids are expected to be a
    constant value.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    yB_size = TEST_PARAMS["yB_size"]

    # Instantiate classes
    dft = make_core(TEST_PARAMS)
    Nx = dft.subgrid_off_step
    Ny = dft.facet_off_step
    assert yB_size % Ny == 0

    # Test with different values and facet offsets
    for val, facet_off in itertools.product(
        [0, 1, 0.1], numpy.arange(-5 * Ny, 5 * Ny // 2, Ny)
    ):

        # Set value at centre of image (might be off-centre for
        # the facet depending on offset)
        facet = numpy.zeros(yB_size)
        facet[yB_size // 2 - facet_off] = val
        prepped = dft.prepare_facet(facet, facet_off, axis=0)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in numpy.arange(0, 10 * Nx, Nx):
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_off, axis=0
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_contrib, facet_off, axis=0
            )
            subgrid = dft.finish_subgrid(subgrid_acc, sg_off, xA_size)

            # Now the entire subgrid should have (close to) a
            # constant value
            numpy.testing.assert_array_almost_equal(subgrid, val / N)


def test_facet_to_subgrid_dft_1d():
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation.
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    yB_size = TEST_PARAMS["yB_size"]

    # Instantiate classes
    dft = make_core(TEST_PARAMS)
    Nx = dft.subgrid_off_step
    Ny = dft.facet_off_step
    assert yB_size % Ny == 0

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
        numpy.arange(-yB_size // 2 + 6 * Ny, yB_size // 2, Ny),
    ):

        # Set sources in facet
        facet = make_facet_from_sources(sources, N, yB_size, [facet_off])

        # We assume all sources are on the facet
        assert numpy.sum(facet) == sum(src[0] for src in sources)
        prepped = dft.prepare_facet(facet, facet_off, axis=0)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_off in [0, Nx, -Nx, N]:
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_off, axis=0
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_contrib, facet_off, axis=0
            )
            subgrid = dft.finish_subgrid(subgrid_acc, sg_off, xA_size)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            numpy.testing.assert_array_almost_equal(subgrid, expected)


def test_facet_to_subgrid_dft_2d():
    """Test facet to subgrid distributed FT primitives against direct
    Fourier transformation implementation -- 2D version
    """

    # Basic layout parameters
    N = TEST_PARAMS["N"]
    xA_size = TEST_PARAMS["xA_size"]
    yB_size = TEST_PARAMS["yB_size"]

    # Instantiate classes
    dft = make_core(TEST_PARAMS)
    Nx = dft.subgrid_off_step
    Ny = dft.facet_off_step
    assert yB_size % Ny == 0

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
        prepped0 = dft.prepare_facet(facet, facet_offs[0], axis=0)
        prepped = dft.prepare_facet(prepped0, facet_offs[1], axis=1)

        # Now generate subgrids at different (valid) subgrid offsets.
        for sg_offs in [[0, 0], [0, Nx], [Nx, 0], [-Nx, -Nx]]:
            subgrid_contrib0 = dft.extract_facet_contrib_to_subgrid(
                prepped, sg_offs[0], axis=0
            )
            subgrid_contrib = dft.extract_facet_contrib_to_subgrid(
                subgrid_contrib0, sg_offs[1], axis=1
            )
            subgrid_acc0 = dft.add_facet_contribution(
                subgrid_contrib, facet_offs[0], axis=0
            )
            subgrid_acc = dft.add_facet_contribution(
                subgrid_acc0, facet_offs[1], axis=1
            )
            subgrid = dft.finish_subgrid(subgrid_acc, sg_offs, xA_size)

            # Now check against DFT
            expected = make_subgrid_from_sources(sources, N, xA_size, sg_offs)
            numpy.testing.assert_array_almost_equal(subgrid, expected)


def test_subgrid_to_facet_basic():
    """Test basic properties of 1D subgrid to facet distributed FT
    primitives for cases where a subgrid is set to a constant value.
    """

    # Basic layout parameters
    xA_size = TEST_PARAMS["xA_size"]
    yB_size = TEST_PARAMS["yB_size"]

    # Instantiate classes
    dft = make_core(TEST_PARAMS)
    Nx = dft.subgrid_off_step
    Ny = dft.facet_off_step
    assert yB_size % Ny == 0
    sg_offs = Nx * numpy.arange(-9, 8)
    facet_offs = numpy.hstack(
        [[-yB_size // 2 + Ny, yB_size // 2], Ny * numpy.arange(-9, 8)]
    )

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for val, sg_off in itertools.product([0, 1, 0.1], sg_offs):

        # Constant-value subgrid
        prepped = dft.prepare_subgrid(
            (val / xA_size) * numpy.ones(xA_size), sg_off
        )

        # Check different facet offsets
        for facet_off in facet_offs:
            extracted = dft.extract_subgrid_contrib_to_facet(
                prepped, facet_off, axis=0
            )
            accumulated = dft.add_subgrid_contribution(
                extracted, sg_off, axis=0
            )
            facet = dft.finish_facet(
                accumulated, facet_off, yB_size, numpy.ones(yB_size), axis=0
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
    xA_size = TEST_PARAMS["xA_size"]
    yB_size = TEST_PARAMS["yB_size"]

    # Instantiate classes
    dft = make_core(TEST_PARAMS)
    Nx = dft.subgrid_off_step
    Ny = dft.facet_off_step
    assert yB_size % Ny == 0

    # Parameters to try
    source_lists = [
        [(1, 0)],
        [(2, 1)],
        [(1, -3)],
        [(-0.1, 5)],
    ]
    sg_offs = Nx * numpy.arange(-9, 8)
    facet_offs = numpy.hstack(
        [[-yB_size // 2 + Ny, yB_size // 2], Ny * numpy.arange(-9, 8)]
    )

    # Test linearity with different values, and start with subgrids at
    # different (valid) subgrid offsets.
    for sources, sg_off in itertools.product(source_lists, sg_offs):

        # Generate subgrid. As we are only filling the grid partially
        # here, we have to scale it.
        subgrid = (
            make_subgrid_from_sources(sources, N, xA_size, [sg_off])
            / xA_size
            * N
        )
        prepped = dft.prepare_subgrid(subgrid, sg_off)

        # Check different facet offsets
        for facet_off in facet_offs:
            extracted = dft.extract_subgrid_contrib_to_facet(
                prepped, facet_off, axis=0
            )
            accumulated = dft.add_subgrid_contribution(
                extracted, sg_off, axis=0
            )
            facet = dft.finish_facet(
                accumulated, facet_off, yB_size, None, axis=0
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
