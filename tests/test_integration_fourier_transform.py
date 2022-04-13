# pylint: disable=redefined-outer-name
"""
End-to-end and integration tests.
"""

import itertools
import logging
from unittest.mock import call, patch

import dask
import numpy
import pytest
from numpy.testing import assert_array_almost_equal

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    SparseFourierTransform,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    fft,
    ifft,
    make_subgrid_and_facet,
)
from src.fourier_transform_dask import (
    cli_parser,
    facet_to_subgrid_2d_method_1,
    facet_to_subgrid_2d_method_2,
    facet_to_subgrid_2d_method_3,
    main,
    run_distributed_fft,
    subgrid_to_facet_algorithm,
)
from tests.test_reference_data.ref_data_2d import (
    EXPECTED_FACET_2D,
    EXPECTED_NONZERO_APPROX_FACET_2D,
    EXPECTED_NONZERO_SUBGRID_2D,
)

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
    return SparseFourierTransform(**TEST_PARAMS)


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
        ([], "1k[1]-512-256"),
        (["--swift_config", "3k[1]-n1536-512"], "3k[1]-n1536-512"),
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
    args = parser.parse_args(["--swift_config", "non-existent-key"])
    expected_message = (
        "Provided argument does not match any swift configuration keys. "
        "Please consult src/swift_configs.py for available options."
    )

    with pytest.raises(KeyError) as error_string:
        main(args)

    # the following is how we can get the error message out of the information
    assert str(error_string.value) == f"'{expected_message}'"


@pytest.mark.parametrize("use_dask", [False, True])
def test_end_to_end_2d_dask(use_dask):
    """
    Test that the 2d algorithm produces the same results with and without dask.
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    # base_arrays_class = BaseArrays(**TEST_PARAMS)
    #
    # # We need to call scipy.special.pro_ang1 function before setting up Dask
    # # context. Detailed information could be found at Jira ORC-1214
    # _ = base_arrays_class.pswf

    if use_dask:
        client = set_up_dask()
    else:
        client = None

    (  # pylint: disable=unused-variable
        result_subgrid,
        result_facet,
        result_approx_subgrid,
        result_approx_facet,
    ) = run_distributed_fft(
        # base_arrays_class,
        TEST_PARAMS,
        to_plot=False,
        use_dask=use_dask,
        client=client,
    )

    # check array shapes
    assert result_subgrid.shape == (6, 6, 188, 188)
    assert result_facet.shape == (4, 4, 256, 256)
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

    # TODO: we need to finish this test
    #  (implement the approx_subgrid tests of it)

    if use_dask:
        tear_down_dask(client)


# this test does not seem to work with the gitlab-ci;
@pytest.mark.skip
@pytest.mark.parametrize("use_dask", [False, True])
def test_end_to_end_2d_dask_logging(use_dask):
    """
    Test that the logged information matches the
    expected listed in test_reference_data/reference_data/README.md

    Reference/expected values generated with numpy.random.seed(123456789)
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    # base_arrays_class = BaseArrays(**TEST_PARAMS)
    # _ = base_arrays_class.pswf

    if use_dask:
        client = set_up_dask()
    else:
        client = None

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
        call(
            "RMSE: %s (image: %s)",
            1.8993993540584405e-17,
            3.5708707856298686e-15,
        ),
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
        run_distributed_fft(
            # base_arrays_class,
            TEST_PARAMS,
            to_plot=False,
            use_dask=use_dask,
            client=client,
        )
        for log_call in expected_log_calls:
            assert log_call in mock_log.call_args_list

    if use_dask:
        tear_down_dask(client)


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
        base_arrays_submit = client.scatter(base_arrays)
    else:
        base_arrays_submit = base_arrays

    subgrid, facet = subgrid_and_facet[0], subgrid_and_facet[1]

    result = tested_function(
        facet, target_distr_fft, base_arrays_submit, use_dask=use_dask
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
        base_arrays_submit = client.scatter(base_arrays)
    else:
        base_arrays_submit = base_arrays

    subgrid, facet = subgrid_and_facet[0], subgrid_and_facet[1]

    result = tested_function(
        subgrid, target_distr_fft, base_arrays_submit, use_dask=use_dask
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
