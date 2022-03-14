import numpy

from src.fourier_transform.fourier_algorithm import pad_mid_on_axis


def test_pad_mid_on_axis_1d():
    """
    1 1 1 --> 0 1 1 1 0
    """
    array = numpy.ones(3)
    desired_size = 5
    result = pad_mid_on_axis(array, desired_size, axis=0)

    assert (result == numpy.array([0, 1, 1, 1, 0])).all()


def test_pad_mid_on_axis_2d_axis0():
    """
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
    result = pad_mid_on_axis(array, desired_size, axis=0)

    assert (result == expected_array).all()


def test_pad_mid_on_axis_2d_axis1():
    """
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
    result = pad_mid_on_axis(array, desired_size, axis=1)

    assert (result == expected_array).all()


def test_pad_mid_on_axis_2d_full():
    """
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
    first_pad = pad_mid_on_axis(array, desired_size, axis=0)
    result = pad_mid_on_axis(first_pad, desired_size, axis=1)

    assert (result == expected_array).all()
