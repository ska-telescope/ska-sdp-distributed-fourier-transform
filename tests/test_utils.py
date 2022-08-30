"""
Unit tests for utils.py functions
"""

import os
import shutil

import dask
import h5py
import numpy
import pytest

from ska_sdp_exec_swiftly.dask_wrapper import set_up_dask, tear_down_dask
from ska_sdp_exec_swiftly.fourier_transform.algorithm_parameters import (
    BaseArrays,
)
from ska_sdp_exec_swiftly.fourier_transform.fourier_algorithm import (
    make_subgrid_and_facet,
    make_subgrid_and_facet_from_hdf5,
)
from ska_sdp_exec_swiftly.generate_hdf5 import generate_data_hdf5
from ska_sdp_exec_swiftly.utils import (
    error_task_facet_to_subgrid_2d,
    error_task_subgrid_to_facet_2d,
    fundamental_errors,
    write_hdf5,
)


def _generate_test_data_hdf5(prefix, use_dask=True):
    """
    For generating hdf5 test data
    """
    numpy.random.seed(123456789)
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

    base_arrays = BaseArrays(**TEST_PARAMS)

    if use_dask:
        client = set_up_dask()
    else:
        client = None
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

    if use_dask:
        tear_down_dask(client)
    return prefix, g_file, fg_file, base_arrays


@pytest.mark.parametrize("use_dask", [False, True])
def test_make_subgrid_and_facet_from_hdf5(use_dask):
    """
    Test that make_subgrid_and_facet_from_hdf5 and make_subgrid_and_facet
    generate the same data
    """

    prefix, g_file, fg_file, base_arrays = _generate_test_data_hdf5(
        prefix="tmpdata2"
    )

    with h5py.File(g_file, "r") as f:
        G = numpy.array(f["G_data"])
    with h5py.File(fg_file, "r") as f:
        FG = numpy.array(f["FG_data"])

    if use_dask:
        client = set_up_dask()

    subgrid, facet = make_subgrid_and_facet_from_hdf5(
        g_file, fg_file, base_arrays, use_dask=use_dask
    )

    subgrid_true, facet_true = make_subgrid_and_facet(
        G, FG, base_arrays, dims=2, use_dask=False
    )

    if use_dask:
        subgrid, facet = dask.compute(subgrid, facet, sync=True)

    assert numpy.array((subgrid == subgrid_true)).all()
    assert numpy.array((facet == facet_true)).all()

    if use_dask:
        tear_down_dask(client)

    # clean up
    if os.path.exists(prefix):
        shutil.rmtree(prefix)


# pylint: disable=too-many-locals
@pytest.mark.parametrize("use_dask", [False, True])
def test_write_hdf5(use_dask):
    """
    Test the correctness of writing subgrid and facets to hdf5
    """

    prefix = "tmpdata3"
    chunksize = 128

    prefix, g_file, fg_file, base_arrays = _generate_test_data_hdf5(
        prefix=prefix,
        use_dask=use_dask,
    )
    with h5py.File(g_file, "r") as f:
        G = numpy.array(f["G_data"])
    with h5py.File(fg_file, "r") as f:
        FG = numpy.array(f["FG_data"])

    if use_dask:
        client = set_up_dask()

    subgrid_true, facet_true = make_subgrid_and_facet(
        G, FG, base_arrays, dims=2, use_dask=False
    )

    approx_subgrid_path = f"{prefix}/approx_G_{G.shape[0]}_{chunksize}.h5"
    approx_facet_path = f"{prefix}/approx_FG_{FG.shape[0]}_{chunksize}.h5"

    p1, p2 = write_hdf5(
        subgrid_true,
        facet_true,
        approx_subgrid_path,
        approx_facet_path,
        base_arrays,
        chunksize,
        chunksize,
        use_dask=use_dask,
    )

    if use_dask:
        p1, p2 = dask.compute(p1, p2, sync=True)

    with h5py.File(approx_subgrid_path, "r") as f:
        approx_G = numpy.array(f["G_data"])
    with h5py.File(approx_facet_path, "r") as f:
        approx_FG = numpy.array(f["FG_data"])

    assert (G == approx_G).all()
    assert (FG == approx_FG).all()

    if use_dask:
        tear_down_dask(client)
    # clean up
    if os.path.exists(prefix):
        shutil.rmtree(prefix)


@pytest.mark.parametrize("use_dask", [False, True])
def test_generate_hdf5(use_dask):
    """
    Test if the correct hdf5 is generated
    """

    prefix = "tmpdata4"
    prefix, g_file, fg_file, base_arrays = _generate_test_data_hdf5(
        prefix=prefix,
        use_dask=use_dask,
    )

    assert os.path.exists(g_file) and os.path.exists(fg_file)

    with h5py.File(g_file, "r") as f:
        G = numpy.array(f["G_data"])
    with h5py.File(fg_file, "r") as f:
        FG = numpy.array(f["FG_data"])

    assert G.shape == (base_arrays.N, base_arrays.N)
    assert FG.shape == (base_arrays.N, base_arrays.N)

    # clean up
    if os.path.exists(prefix):
        shutil.rmtree(prefix)


@pytest.mark.parametrize("use_dask", [False, True])
def test_errors_task(use_dask):
    """
    For testing the correctness of calculating subgrid and facets error
     terms using dask and serial
    """

    nsubgrid = 3
    xA_size = 256

    numpy.random.seed(123456789)
    approx_data = [
        [numpy.random.random((xA_size, xA_size)) for i in range(nsubgrid)]
        for i in range(nsubgrid)
    ]
    true_data = [
        [numpy.random.random((xA_size, xA_size)) for i in range(nsubgrid)]
        for i in range(nsubgrid)
    ]

    if use_dask:
        client = set_up_dask()

    res1 = fundamental_errors(
        approx_data,
        nsubgrid,
        true_data,
        error_task_facet_to_subgrid_2d,
        use_dask=use_dask,
    )
    res2 = fundamental_errors(
        approx_data,
        nsubgrid,
        true_data,
        error_task_subgrid_to_facet_2d,
        use_dask=use_dask,
    )

    if use_dask:
        res1, res2 = dask.compute(res1, res2)
        tear_down_dask(client)

    true_res1 = (0.40801243491030215, 104.45118333703734)
    true_res2 = (0.0015937985738683678, 0.40801243491030215)

    assert numpy.isclose(res1[0], true_res1[0])
    assert numpy.isclose(res1[1], true_res1[1])

    assert numpy.isclose(res2[0], true_res2[0])
    assert numpy.isclose(res2[1], true_res2[1])
