import numpy
import pytest

from src.fourier_transform.fourier_algorithm import (
    pad_mid,
    extract_mid,
    fft,
    ifft,
    coordinates,
    broadcast,
    create_slice,
)


def test_pad_mid_1d():
    """
    perform operation on 1D array

    1 1 1 --> 0 1 1 1 0
    """
    array = numpy.ones(3)
    desired_size = 5
    result = pad_mid(array, desired_size, axis=0)

    assert (result == numpy.array([0, 1, 1, 1, 0])).all()


def test_pad_mid_2d_axis0():
    """
    perform operation for axis=0

                0 0 0
    1 1 1       1 1 1
    1 1 1  -->  1 1 1
    1 1 1       1 1 1
                0 0 0
    """
    array = numpy.ones((3, 3))
    desired_size = 5
    expected_array = numpy.array(
        [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]]
    )
    result = pad_mid(array, desired_size, axis=0)

    assert (result == expected_array).all()


def test_pad_mid_2d_axis1():
    """
    perform operation for axis=1

    1 1 1       0 1 1 1 0
    1 1 1  -->  0 1 1 1 0
    1 1 1       0 1 1 1 0
    """
    array = numpy.ones((3, 3))
    desired_size = 5
    expected_array = numpy.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ]
    )
    result = pad_mid(array, desired_size, axis=1)

    assert (result == expected_array).all()


def test_pad_mid_2d_axis01():
    """
    perform operation for axis=0 and axis=1

                0 0 0 0 0
    1 1 1       0 1 1 1 0
    1 1 1  -->  0 1 1 1 0
    1 1 1       0 1 1 1 0
                0 0 0 0 0
    """
    array = numpy.ones((3, 3))
    desired_size = 5
    expected_array = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    first_pad = pad_mid(array, desired_size, axis=0)
    result = pad_mid(first_pad, desired_size, axis=1)

    assert (result == expected_array).all()


def test_extract_mid_1d():
    """
    perform operation on 1D array

    if new size can be evenly extracted from original size:
    7 -> 5
    x y y y y y x --> y y y y y

    if new size doesn't allow for even extraction from the middle,
    then the middle is extracted plus the one element just before the middle
    7 -> 4
    x x y y y x x --> x y y y
    """
    full_array = numpy.array([0, 1, 2, 3, 4, 5, 6])
    desired_size = 5
    result = extract_mid(full_array, desired_size, axis=0)

    assert (result == numpy.array([1, 2, 3, 4, 5])).all()

    desired_size = 4
    result = extract_mid(full_array, desired_size, axis=0)

    assert (result == numpy.array([1, 2, 3, 4])).all()


def test_extract_mid_2d_axis0():
    """
    perform operation for axis=0

    if new size can be evenly extracted from original size:
    3 -> 1
    x x x x
    x y y x --> x y y x
    x x x x

    if new size doesn't allow for even extraction from the middle,
    then the middle is extracted plus the one element just BEFORE the middle
    (AXIS=0 only)
    3 -> 2
    x x x x     x x x x
    x y y x --> x y y x
    x x x x
    """
    full_array = numpy.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    desired_size = 1
    expected_array = numpy.array([[4, 5, 6, 7]])
    result = extract_mid(full_array, desired_size, axis=0)
    assert (result == expected_array).all()

    desired_size = 2
    expected_array = numpy.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    result = extract_mid(full_array, desired_size, axis=0)

    assert (result == expected_array).all()


def test_extract_mid_2d_axis1():
    """
    perform operation for axis=1

    if new size can be evenly extracted from original size:
    4 -> 2
    x x x x     x x
    x y y x --> y y
    x x x x     x x

    if new size doesn't allow for even extraction from the middle,
    then the middle is extracted plus the one element just AFTER the middle
    (AXIS=1 only)
    4 -> 3
    x x x x     x x x
    x y y x --> y y x
    x x x x     x x x
    """
    full_array = numpy.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    desired_size = 2
    expected_array = numpy.array([[1, 2], [5, 6], [9, 10]])
    result = extract_mid(full_array, desired_size, axis=1)

    assert (result == expected_array).all()

    desired_size = 3
    expected_array = numpy.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
    result = extract_mid(full_array, desired_size, axis=1)

    assert (result == expected_array).all()


def test_extract_mid_2d_axis01():
    """
    perform operation for axis=0 and axis=1

    square matrix (input and output too):
    5x5 -> 3x3
    x x x x x
    x y y y x       y y y
    x y y y x  -->  y y y
    x y y y x       y y y
    x x x x x


    5x5 -> 3x2
    x x x x x
    x y y z x       y y
    x y y z x  -->  y y
    x y y z x       y y
    x x x x x
    """
    full_array = numpy.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    )
    desired_size = 3
    expected_array = numpy.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]])
    result = extract_mid(
        extract_mid(full_array, desired_size, axis=0), desired_size, axis=1
    )

    assert (result == expected_array).all()

    desired_size_axis0 = 3
    desired_size_axis1 = 2
    expected_array = numpy.array([[6, 7], [11, 12], [16, 17]])
    result = extract_mid(
        extract_mid(full_array, desired_size_axis0, axis=0),
        desired_size_axis1,
        axis=1,
    )

    assert (result == expected_array).all()


def test_fft_1d():
    """
    FFT of a 1D array (== axis=0)

    input array: -->  fft (complex):
    1 1 1 1 1         0 0 5 0 0
    """
    array = numpy.ones(5)
    result = fft(array, axis=0)
    assert result.dtype == complex
    assert (result == numpy.array([0, 0, 5, 0, 0], dtype=complex)).all()


def test_fft_2d_axis0():
    """
    FFt along axis=0

    input array: -->  fft (complex):
    1 1 1 1 1         0 0 0 0 0
    1 1 1 1 1         3 3 3 3 3
    1 1 1 1 1         0 0 0 0 0
    """
    array = numpy.ones((3, 5))
    result = fft(array, axis=0)
    assert result.dtype == complex
    assert (
        result[numpy.where(result != 0)] == numpy.array([3, 3, 3, 3, 3], dtype=complex)
    ).all()


def test_fft_2d_axis1():
    """
    FFT along axis=1

    input array: -->  fft (complex):
    1 1 1 1 1         0 0 5 0 0
    1 1 1 1 1         0 0 5 0 0
    1 1 1 1 1         0 0 5 0 0
    """
    array = numpy.ones((3, 5))
    result = fft(array, axis=1)
    assert result.dtype == complex
    assert (
        result[numpy.where(result != 0)] == numpy.array([[5], [5], [5]], dtype=complex)
    ).all()


def test_fft_2d_axis01():
    """
    FFT along axis=0 and axis=1

    input array: -->  fft (complex):
    1 1 1 1 1         0 0 0 0 0
    1 1 1 1 1         0 0 15 0 0
    1 1 1 1 1         0 0 0 0 0
    """
    array = numpy.ones((3, 5))
    result = fft(fft(array, axis=0), axis=1)
    assert result.dtype == complex
    assert (result[numpy.where(result != 0)] == numpy.array([15], dtype=complex)).all()


def test_ifft_1d():
    """
    iFFT of a 1D array (== axis=0)
    """
    result = ifft(numpy.array([0, 0, 5, 0, 0], dtype=complex), axis=0)
    assert (result == numpy.ones(5)).all()


def test_ifft_2d_axis0():
    """
    iFFT along axis=0
    """
    array = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=complex,
    )

    result = ifft(array, axis=0)

    assert (
        result == numpy.array([[0, 0, 5, 0, 0], [0, 0, 5, 0, 0], [0, 0, 5, 0, 0]])
    ).all()


def test_ifft_2d_axis1():
    """
    iFFT along axis=1
    """
    array = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=complex,
    )

    result = ifft(array, axis=1)

    assert (
        result == numpy.array([[0, 0, 0, 0, 0], [3, 3, 3, 3, 3], [0, 0, 0, 0, 0]])
    ).all()


def test_ifft_2d_axis01():
    """
    iFFT along axis=0 and axis=1
    """
    array = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=complex,
    )
    result = ifft(ifft(array, axis=0), axis=1)

    assert (result == numpy.ones((3, 5))).all()


@pytest.mark.parametrize(
    "n, minimum, maximum",
    [
        (8, -0.5, 0.375),
        (10, -0.5, 0.4),
        (23, -0.47826087, 0.47826087),
        (50, -0.5, 0.48),
        (100, -0.5, 0.49),
        (1000, -0.5, 0.499),
    ],
)
def test_coordinates(n, minimum, maximum):
    """
    Test values are chosen to illustrate how min and max
    values in the array change depending on the array length.
    """
    result = coordinates(n)

    assert len(result) == n
    assert result[0].round(8) == round(minimum, 8)
    assert result[n // 2] == 0.0
    assert result[-1].round(8) == round(maximum, 8)


@pytest.mark.parametrize(
    "dims, axis, expected_shape",
    [
        (0, 0, (10, 10)),
        (1, 0, (10, 10)),
        (2, 0, (10, 1, 10)),
        (3, 0, (10, 1, 1, 10)),
        (4, 0, (10, 1, 1, 1, 10)),
        (0, 1, (10, 10)),
        (1, 1, (1, 10, 10)),
        (2, 1, (1, 10, 10)),
        (3, 1, (1, 10, 1, 10)),
        (4, 1, (1, 10, 1, 1, 10)),
        (0, 2, (10, 10)),
        (1, 2, (1, 10, 10)),
        (2, 2, (1, 1, 10, 10)),
        (3, 2, (1, 1, 10, 10)),
        (4, 2, (1, 1, 10, 1, 10)),
        (0, 3, (10, 10)),
        (1, 3, (1, 10, 10)),
        (2, 3, (1, 1, 10, 10)),
        (3, 3, (1, 1, 1, 10, 10)),
        (4, 3, (1, 1, 1, 10, 10)),
        (5, 3, (1, 1, 1, 10, 1, 10)),
        (0, 3, (10, 10)),
    ],
)
def test_broadcast(dims, axis, expected_shape):
    """
    Provide a large set of cases to indicate how
    the shape of the input array changes
    with input dims-axis combinations.
    """
    array = numpy.ones((10, 10))
    result = broadcast(array, dims, axis)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "dims, axis",
    [
        (1, (0, 1)),
        (2, (0, 2)),
        (3, (1, 1)),
        ((2, 4), 4),
        ("str", (3, 4)),
    ],
)
def test_broadcast_raises_error(dims, axis):
    """
    ValueError is raised when either dims or axis is not an integer.
    See docstring and test for create_slice.
    """
    with pytest.raises(ValueError):
        broadcast(numpy.ones((10, 10)), dims, axis)


@pytest.mark.parametrize(
    "dims, axis, expected_tuple",
    [
        (0, 0, ()),  # if dims is 0, result is always an empty tuple
        (1, 0, (6,)),  # range(1) --> 0, which equals to axis -> use axis_value
        (1, 1, (2,)),  # range(1) --> 0, which doesn't equal to axis -> use fill_value
        (3, 2, (2, 2, 6)),  # axis=2 (3rd value in tuple) is axis_val
        (6, 3, (2, 2, 2, 6, 2, 2)),
    ],
)
def test_create_slice(dims, axis, expected_tuple):
    fill_val = 2
    axis_val = 6
    result = create_slice(fill_val, axis_val, dims, axis)
    assert result == expected_tuple


@pytest.mark.parametrize(
    "dims, axis", [(5, (0, 2)), ((2, 3), 4), ((2, 2), (0, 1)), ("bla", 5), (3, "bla")]
)
def test_create_slice_raises_error(dims, axis):
    """
    Only integers of dims and axis are allowed.
    While axis could be other things too, that would not have an
    effect, since in that case axis would never be in range(dims),
    hence we do not allow it in the code.
    """
    with pytest.raises(ValueError):
        create_slice(2, 6, dims, axis)
