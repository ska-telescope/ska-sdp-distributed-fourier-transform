import numpy
import pytest

from src.fourier_transform.fourier_algorithm import pad_mid_a, pad_mid, generate_mask


@pytest.mark.parametrize("axis, expected_shape", [(0, (6, 4)), (1, (2, 6))])
def test_pad_mid_a_2d(axis, expected_shape):
    a = numpy.zeros((2, 4), dtype=float)
    size_to_pad = 6
    result = pad_mid_a(a, size_to_pad, axis)

    assert result.shape == expected_shape


def test_pad_mid_a_1d():
    """
    pad_mid_a with axis=0, should behave the same way as pad_mid for 1D arrays
    """
    a = numpy.zeros((2,), dtype=float)
    size_to_pad = 6
    result = pad_mid_a(a, size_to_pad, 0)

    assert result.shape == (6,)


def test_pad_mid_a_2d_check_padding():
    """
    2 1 1 2
    2 1 1 2
    2 1 1 2
    2 1 1 2
    --> padded at axis=0:
    0 0 0 0
    2 1 1 2
    2 1 1 2
    2 1 1 2
    2 1 1 2
    0 0 0 0
    """
    a = numpy.zeros((4, 4), dtype=float)
    a[:, 1:3] = 1
    a[:, 0] = 2
    a[:, 3] = 2
    size_to_pad = 6
    result = pad_mid_a(a, size_to_pad, 0)

    assert (result[1:5, :] == a).all()
    assert (result[0, :] == 0).all()
    assert (result[-1, :] == 0).all()


def test_pad_mid_1d():
    # if a is 2D, pad_mid expects both dims to be the same size
    a = numpy.zeros((3,), dtype=float)
    size_to_pad = 6
    result = pad_mid(a, size_to_pad)

    assert result.shape == (6,)


def test_pad_mid_2d():
    # if a is 2D, pad_mid expects both dims to be the same size
    a = numpy.zeros((4, 4), dtype=float)
    size_to_pad = 6
    result = pad_mid(a, size_to_pad)

    assert result.shape == (6, 6)


def test_pad_mid_2d_fail():
    # if a is 2D, pad_mid expects both dims to be the same size
    # else it raises an assertion error
    a = numpy.zeros((2, 4), dtype=float)
    size_to_pad = 6

    with pytest.raises(AssertionError):
        pad_mid(a, size_to_pad)


def test_generate_mask():
    """
    Resulting mask of this test:

    array([[1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0]])
    """
    n = 10
    ndata = 3
    true_size = 5
    offset = numpy.array([0, 5, 7])

    expected_mask = numpy.array(
        [[1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0]]
    )
    result = generate_mask(n, ndata, true_size, offset)

    assert result.shape == (ndata, true_size)
    assert (result == expected_mask).all()
