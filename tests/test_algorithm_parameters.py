import numpy
import pytest

from src.fourier_transform.algorithm_parameters import (
    ConstantParams,
    ConstantArrays,
)

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


def test_constant_params_fundamental():
    """
    Input dictionary values are correctly assigned to
    fundamental attributes of the class.
    """
    result = ConstantParams(**TEST_PARAMS)

    for k, v in TEST_PARAMS.items():
        assert result.__getattribute__(k) == v


def test_constant_params_derived():
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

    result = ConstantParams(**TEST_PARAMS)

    for k, v in expected_derived.items():
        assert result.__getattribute__(k) == v


def test_constant_params_check_params():
    """
    ConstantParams.check_params is called as part of __init__
    It raises a ValueError if a certain condition doesn't apply,
    which can be achieved by slightly altering, e.g. N
    """
    new_params = TEST_PARAMS.copy()
    new_params["N"] = 1050

    with pytest.raises(ValueError):
        ConstantParams(**new_params)


@pytest.mark.parametrize(
    "attribute, expected_array",
    [
        ("facet_off", [0, 256, 512, 768]),
        ("subgrid_off", [4, 192, 380, 568, 756, 944]),
    ],
)
def test_constant_arrays_offsets(attribute, expected_array):
    """
    Offsets are correctly calculated using input parameters.
    """
    array_class = ConstantArrays(**TEST_PARAMS)

    result = array_class.__getattribute__(attribute)
    assert (result == numpy.array(expected_array)).all()


def test_constant_arrays_generate_mask():
    """
    Using subgrid_off and xA_size and nsubgrid, as would
    the code with the values specified by TEST_PARAMS
    """
    array_class = ConstantArrays(**TEST_PARAMS)

    mask_dim1 = 6
    mask_dim2 = 188
    offsets = [4, 192, 380, 568, 756, 944]

    mask = array_class._generate_mask(mask_dim1, mask_dim2, offsets)
    assert mask.shape == (mask_dim1, mask_dim2)
    assert (mask[0, :52] == 0.0).all()
    assert (mask[5, -52:] == 0.0).all()
    assert (mask[1:5, :] == 1.0).all()
    assert (mask[0, 53:] == 1.0).all()
    assert (mask[5, :-52] == 1.0).all()


def test_constant_arrays_generate_mask_index_error():
    """
    IndexError is raised if the length of offsets
    is smaller than mask_dim1.
    """
    array_class = ConstantArrays(**TEST_PARAMS)

    mask_dim1 = 6
    mask_dim2 = 188
    offsets = [4, 192, 380, 568, 756]

    with pytest.raises(IndexError):
        array_class._generate_mask(mask_dim1, mask_dim2, offsets)


def test_constant_arrays_pure_arrays():
    """
    Fb, Fn, facet_m0_trunc, pswf are complicated arrays and their code is
    based on pure calculations, therefore I decided not to test
    their actual values, only that the code calculating them doesn't break
    when the class is instantiated with correct parameters.
    """
    array_class = ConstantArrays(**TEST_PARAMS)
    fb = array_class.Fb
    fn = array_class.Fn
    facet_m0_trunc = array_class.facet_m0_trunc
    pswf = array_class.pswf
