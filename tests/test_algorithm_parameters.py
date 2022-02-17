import numpy
import pytest

from src.fourier_transform.algorithm_parameters import ConstantParams, ConstantArrays

INPUT_PARAMS = {
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

EXPECTED_DERIVATES = {
    "xM_yP_size": 128,
    "xM_yN_size": 80,
    "xMxN_yP_size": 150,
    "xN_yP_size": 22,
    "nsubgrid": 6,
    "nfacet": 4
}


def test_constant_params():
    """
    class ConstantParams correctly calculates the derived values
    and stores the fundamental ones.
    """
    result = ConstantParams(**INPUT_PARAMS)

    for attr in INPUT_PARAMS.keys():
        assert result.__getattribute__(attr) == INPUT_PARAMS[attr]

    for attr in EXPECTED_DERIVATES.keys():
        assert result.__getattribute__(attr) == EXPECTED_DERIVATES[attr]


def test_constant_params_missing():
    """
    class ConstantParams raises a KeyError when
    any of the fundamental params is missing
    """
    for key in INPUT_PARAMS.keys():
        new_pars = INPUT_PARAMS.copy()
        new_pars.pop(key)

        with pytest.raises(KeyError):
            ConstantParams(**new_pars)


def test_constant_params_string():
    """
    String representation of ConstantParams class contains
    all the fundamental and derived values.
    """
    dict_string = str(ConstantParams(**INPUT_PARAMS))
    result = dict_string.split("\n")
    for (key, value), (key2, value2) in zip(INPUT_PARAMS.items(), EXPECTED_DERIVATES.items()):
        assert f"{key} = {value}" in result
        assert f"{key2} = {value2}" in result


def test_constant_arrays():
    """
    class ConstantArrays correctly calculates its properties

    TODO:
        Fb, Fn, facet_m0_trunc, pswf
    """
    arrays_class = ConstantArrays(**INPUT_PARAMS)

    assert (arrays_class.facet_off == numpy.array([0, 256, 256*2, 256*3])).all()
    assert (arrays_class.subgrid_off == numpy.array([0+4, 188+4, 188*2+4, 188*3+4, 188*4+4, 188*5+4])).all()
