# pylint: disable=redefined-outer-name, unused-variable
"""
End-to-end and integration tests.
"""

import itertools
import logging
import os
import shutil

import dask
import h5py
import numpy
import pytest
from numpy.testing import assert_array_almost_equal

from ska_sdp_exec_swiftly.dask_wrapper import set_up_dask, tear_down_dask
from ska_sdp_exec_swiftly.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from ska_sdp_exec_swiftly.fourier_transform.fourier_algorithm import (
    fft,
    ifft,
    make_subgrid_and_facet,
)
from ska_sdp_exec_swiftly.fourier_transform_dask import (
    cli_parser,
    facet_to_subgrid_2d_method_1,
    facet_to_subgrid_2d_method_2,
    facet_to_subgrid_2d_method_3,
    main,
    run_distributed_fft,
    subgrid_to_facet_algorithm,
)
from ska_sdp_exec_swiftly.generate_hdf5 import generate_data_hdf5

log = logging.getLogger("fourier-logger")
log.setLevel(logging.WARNING)

TEST_PARAMS = {
    "W": 13.25,
    "fov": 0.75,
    "N": 1024,
    "Nx": 4,
    "yB_size": 256,
    "yN_size": 320,
    "yP_size": 512,
    "xA_size": 188,
    "xM_size": 256,
}


def _check_difference(calculated, original, size):
    err_mean = 0
    err_mean_img = 0
    for i0, i1 in itertools.product(range(size), range(size)):
        err_mean += (
            numpy.abs(calculated[i0, i1] - original[i0, i1]) ** 2 / size**2
        )
        err_mean_img += (
            numpy.abs(
                fft(fft(calculated[i0, i1] - original[i0, i1], axis=0), axis=1)
            )
            ** 2
            / size**2
        )
    return err_mean, err_mean_img


@pytest.fixture(scope="module")
def target_distr_fft():
    """
    Pytest fixture for instantiated SparseFourierTransform
    """
    return StreamingDistributedFFT(**TEST_PARAMS)


@pytest.fixture(scope="module")
def base_arrays():
    """
    Pytest fixture for instantiated SparseFourierTransform
    """
    return BaseArrays(**TEST_PARAMS)


@pytest.fixture(scope="module")
def subgrid_and_facet(target_distr_fft, base_arrays):
    """
    Pytest fixture for generating subgrid and facet array for tests
    """
    fg = numpy.zeros((target_distr_fft.N, target_distr_fft.N))
    fg[252, 252] = 1
    g = ifft(ifft(fg, axis=0), axis=1)

    subgrid, facet = make_subgrid_and_facet(
        g,
        fg,
        base_arrays,
        dims=2,
        use_dask=False,
    )
    return subgrid, facet


@pytest.mark.parametrize(
    "args, expected_config_key",
    [
        ([], "1k[1]-n512-256"),
        (["--swift_config", "3k[1]-n1536-512"], "3k[1]-n1536-512"),
        (
            ["--swift_config", "1k[1]-n512-256,3k[1]-n1536-512"],
            "1k[1]-n512-256,3k[1]-n1536-512",
        ),
    ],
)
def test_cli_parser(args, expected_config_key):
    """
    cli_parser correctly parses command line arguments
    and uses defaults.
    """
    parser = cli_parser()
    result = parser.parse_args(args)
    assert result.swift_config == expected_config_key


def test_main_wrong_arg():
    """
    main raises KeyError with correct message,
    when the wrong swift_config key is provided.
    """
    parser = cli_parser()
    args = parser.parse_args(
        ["--swift_config", "1k[1]-n512-256,non-existent-key"]
    )
    expected_message = (
        "Provided argument (non-existent-key) does not match any "
        "swift configuration keys. Please consult src/swift_configs.py "
        "for available options."
    )

    with pytest.raises(KeyError) as error_string:
        main(args)

    # the following is how we can get the error message out of the information
    assert str(error_string.value) == f"'{expected_message}'"


def test_end_to_end_2d_dask():
    """
    Test that the 2d algorithm produces the same results
    with and without dask.

    TODO: we need to finish this test
        (implement the approx_subgrid tests of it)
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    client = set_up_dask()

    (  # pylint: disable=unused-variable
        result_subgrid,
        result_facet,
        result_approx_subgrid,
        result_approx_facet,
    ) = run_distributed_fft(TEST_PARAMS, to_plot=False, use_dask=False)

    numpy.random.seed(123456789)

    (  # pylint: disable=unused-variable
        result_subgrid_dask,
        result_facet_dask,
        result_approx_subgrid_dask,
        result_approx_facet_dask,
    ) = run_distributed_fft(
        TEST_PARAMS,
        to_plot=False,
        use_dask=True,
        client=client,
    )

    # check arrays
    assert_array_almost_equal(
        result_subgrid,
        result_subgrid_dask,
    )
    assert_array_almost_equal(
        result_facet,
        result_facet_dask,
    )
    assert_array_almost_equal(
        result_approx_subgrid,
        result_approx_subgrid_dask,
    )
    assert_array_almost_equal(
        result_approx_facet,
        result_approx_facet_dask,
    )

    tear_down_dask(client)


# pylint: disable=too-many-locals
@pytest.mark.parametrize("use_dask", [True, False])
def test_end_to_end_2d_dask_hdf5(use_dask):
    """
    Test that the 2d algorithm produces the same results with dask and hdf5.
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    if use_dask:
        client = set_up_dask()
    else:
        client = None

    prefix = "tmpdata"
    chunksize = 128
    g_file = f"{prefix}/G_{TEST_PARAMS['N']}_{chunksize}.h5"
    fg_file = f"{prefix}/FG_{TEST_PARAMS['N']}_{chunksize}.h5"

    if not os.path.exists(prefix):
        os.makedirs(prefix)
    else:
        shutil.rmtree(prefix)
        os.makedirs(prefix)

    generate_data_hdf5(
        TEST_PARAMS["N"],
        G_2_path=g_file,
        FG_2_path=fg_file,
        chunksize_G=chunksize,
        chunksize_FG=chunksize,
        client=client,
    )

    (
        G_2_file,
        FG_2_file,
        approx_G_2_file,
        approx_FG_2_file,
    ) = run_distributed_fft(
        TEST_PARAMS,
        to_plot=False,
        use_dask=use_dask,
        client=client,
        use_hdf5=True,
        hdf5_prefix=prefix,
        hdf5_chunksize=[chunksize, chunksize],
    )

    if use_dask:
        tear_down_dask(client)

    # compare hdf5
    with h5py.File(G_2_file, "r") as f:
        G = numpy.array(f["G_data"])
    with h5py.File(approx_G_2_file, "r") as f:
        AG = numpy.array(f["G_data"])
    with h5py.File(FG_2_file, "r") as f:
        FG = numpy.array(f["FG_data"])
    with h5py.File(approx_FG_2_file, "r") as f:
        AFG = numpy.array(f["FG_data"])

    # clean up
    if os.path.exists(prefix):
        shutil.rmtree(prefix)

    error_G = numpy.std(numpy.abs(G - AG))
    assert error_G < 3e-08

    error_FG = numpy.std(numpy.abs(FG - AFG))
    assert error_FG < 5e-05


@pytest.mark.parametrize(
    "use_dask, tested_function",
    [
        (False, facet_to_subgrid_2d_method_1),
        (False, facet_to_subgrid_2d_method_2),
        (False, facet_to_subgrid_2d_method_3),
        (True, facet_to_subgrid_2d_method_1),
        (True, facet_to_subgrid_2d_method_2),
        (True, facet_to_subgrid_2d_method_3),
    ],
)
def test_facet_to_subgrid_methods(
    use_dask, tested_function, target_distr_fft, base_arrays, subgrid_and_facet
):
    """
    Integration test for facet->subgrid algorithm.
    Three versions are provided (see their docstrings), but they
    all do the same, just iterate in a different order.

    Here, we test all, both with and without Dask.
    The input facet array is always the same.

    We check that the difference between the original subgrid array
    and the output approximate subgrid array (result) is negligible.
    """
    if use_dask:
        client = set_up_dask()

    subgrid, facet = subgrid_and_facet[0], subgrid_and_facet[1]

    result = tested_function(
        facet, target_distr_fft, base_arrays, use_dask=use_dask
    )
    if use_dask:
        result = dask.compute(result, sync=True)[0]
        result = numpy.array(result)

    assert result.shape == subgrid.shape

    numpy.testing.assert_array_almost_equal(
        abs(result), abs(subgrid), decimal=15
    )

    error_mean, error_mean_img = _check_difference(
        result, subgrid, target_distr_fft.nsubgrid
    )
    assert (error_mean < 1e-16).all()
    assert (error_mean_img < 1e-16).all()

    if use_dask:
        tear_down_dask(client)


@pytest.mark.parametrize(
    "use_dask,tested_function",
    [
        (False, subgrid_to_facet_algorithm),
        (True, subgrid_to_facet_algorithm),
    ],
)
def test_subgrid_to_facet(
    use_dask, tested_function, target_distr_fft, base_arrays, subgrid_and_facet
):
    """
    Integration test for subgrid->facet algorithm.

    We check that the difference between the original facet array
    and the output approximate facet array (result) is negligible.

    Due to precision errors, the two arrays (result and facet)
    will only be equal to a precision of 1e-7 (<1e-8).
    TODO: need to investigate if above is true, and why is the
      other direction much more precise!
    """
    if use_dask:
        client = set_up_dask()

    subgrid, facet = subgrid_and_facet[0], subgrid_and_facet[1]

    result = tested_function(
        subgrid, target_distr_fft, base_arrays, use_dask=use_dask
    )

    if use_dask:
        result = dask.compute(result, sync=True)[0]
        result = numpy.array(result)

    assert result.shape == facet.shape

    numpy.testing.assert_array_almost_equal(abs(result), abs(facet), decimal=7)

    error_mean, error_mean_img = _check_difference(
        result, facet, target_distr_fft.nfacet
    )
    assert (error_mean < 1e-14).all()
    assert (error_mean_img < 1e-14).all()

    if use_dask:
        tear_down_dask(client)
