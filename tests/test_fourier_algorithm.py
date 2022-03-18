import numpy

from src.fourier_transform.fourier_algorithm import (
    pad_mid,
    extract_mid,
    fft,
    ifft,
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
