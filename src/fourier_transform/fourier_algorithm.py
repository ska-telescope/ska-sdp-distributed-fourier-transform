"""
Distributed Fourier Transform Module.
Included are a list of base functions that are used across the code.
"""
import itertools

import numpy

from src.fourier_transform.dask_wrapper import dask_wrapper


def create_slice(fill_val, axis_val, dims, axis):
    """
    Create a tuple of length = dims.

    Elements of the tuple:
        fill_val if axis != dim_index;
        axis_val if axis == dim_index,
        where dim_index is each value in range(dims)

    See test for examples.

    :param fill_val: value to use for dimensions where dim != axis
    :param axis_val: value to use for dimensions where dim == axis
    :param dims: length of tuple to be produced
                 (i.e. number of dimensions); int
    :param axis: axis (index) along which axis_val to be used; int

    :return: tuple of length dims
    """
    # pylint: disable=consider-using-generator
    # TODO: pylint's suggestion of using a generator should be investigated
    if not isinstance(axis, int) or not isinstance(dims, int):
        raise ValueError(
            "create_slice: axis and dims values have to be integers."
        )

    return tuple([axis_val if i == axis else fill_val for i in range(dims)])


def broadcast(a, dims, axis):
    """
    Stretch input array to shape determined by the dims and axis values.
    See tests for examples of how the shape of the input array will change
    depending on what dims-axis combination is given

    :param a: input numpy ndarray
    :param dims: dimensions to broadcast ("stretch") input array to; int
    :param axis: axis along which the new dimension(s) should be added; int

    :return: array with new shape
    """
    return a[create_slice(numpy.newaxis, slice(None), dims, axis)]


def pad_mid(a, n, axis):
    """
    Pad an array to a desired size with zeros at a given axis.
    (Surround the middle with zeros till it reaches the given size)

    :param a: numpy array to be padded
    :param n: size to be padded to (desired size)
    :param axis: axis along which to pad

    :return: padded numpy array
    """
    n0 = a.shape[axis]
    if n == n0:
        return a
    pad = create_slice(
        (0, 0),
        (n // 2 - n0 // 2, (n + 1) // 2 - (n0 + 1) // 2),
        len(a.shape),
        axis,
    )
    return numpy.pad(a, pad, mode="constant", constant_values=0.0)


def extract_mid(a, n, axis):
    """
    Extract a section from middle of a map (array) along a given axis.
    This is the reverse operation to pad.

    :param a: numpy array from which to extract
    :param n: size of section
    :param axis: axis along which to extract (int: 0, 1)

    :return: extracted numpy array
    """
    assert n <= a.shape[axis]
    cx = a.shape[axis] // 2
    if n % 2 != 0:
        slc = slice(cx - n // 2, cx + n // 2 + 1)
    else:
        slc = slice(cx - n // 2, cx + n // 2)
    return a[create_slice(slice(None), slc, len(a.shape), axis)]


def fft(a, axis):
    """
    Fourier transformation from image to grid space, along a given axis.

    :param a: numpy array, 1D or 2D (image in `lm` coordinate space)
    :param axis: int; axes over which to calculate

    :return: numpy array (`uv` grid)
    """
    return numpy.fft.fftshift(
        numpy.fft.fft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def ifft(a, axis):
    """
    Fourier transformation from grid to image space, along a given axis.
    (inverse Fourier transform)

    :param a: numpy array, 1D or 2D (`uv` grid to transform)
    :param axis: int; axes over which to calculate

    :return: numpy array (an image in `lm` coordinate space)
    """
    return numpy.fft.fftshift(
        numpy.fft.ifft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def coordinates(n):
    """
    Generate a 1D array with length n,
    which spans [-0.5,0.5] with 0 at position n/2.
    See also docs for numpy.mgrid.

    :param n: length of array to be generated
    :return: 1D numpy array
    """
    n2 = n // 2
    if n % 2 == 0:
        return numpy.mgrid[-n2:n2] / n

    return numpy.mgrid[-n2 : n2 + 1] / n


@dask_wrapper
def _ith_subgrid_facet_element(
    true_image, offset_i, true_usable_size, mask_element, axis=(0, 1), **kwargs
):  # pylint: disable=unused-argument
    """
    Calculate a single facet or subgrid element.

    :param true_image: true image, G (1D or 2D)
    :param offset_i: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet
    :param mask_element: an element of subgrid_A or facet_B (masks)
    :param axis: axis (0, 1, or a tuple of both)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    if isinstance(axis, int):
        extracted = extract_mid(
            numpy.roll(true_image, offset_i, axis), true_usable_size, axis
        )

    if isinstance(axis, tuple) and len(axis) == 2:
        extracted = extract_mid(
            extract_mid(
                numpy.roll(true_image, offset_i, axis),
                true_usable_size,
                axis[0],
            ),
            true_usable_size,
            axis[1],
        )

    result = mask_element * extracted
    return result


# TODO: I (GH) tried adding this function as method to the class
#   separate for subgrid and facet, but when calling dask on it
#   the computation becomes extremely slow and my laptop cannot handle it.
#   This suggests that something wasn't right and the dask setup wasn't ideal
#   hence I left these here as a separate function, and not part of the class.
# pylint: disable=too-many-arguments
def make_subgrid_and_facet(
    G,
    FG,
    base_arrays,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param base_arrays: BaseArrays class object
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty(
            (base_arrays.nsubgrid, base_arrays.xA_size), dtype=complex
        )
        facet = numpy.empty(
            (base_arrays.nfacet, base_arrays.yB_size), dtype=complex
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(base_arrays.nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G,
                -base_arrays.subgrid_off[i],
                base_arrays.xA_size,
                base_arrays.subgrid_A[i],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(base_arrays.nfacet):
            facet[j] = _ith_subgrid_facet_element(
                FG,
                -base_arrays.facet_off[j],
                base_arrays.yB_size,
                base_arrays.facet_B[j],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty(
            (
                base_arrays.nsubgrid,
                base_arrays.nsubgrid,
                base_arrays.xA_size,
                base_arrays.xA_size,
            ),
            dtype=complex,
        )
        facet = numpy.empty(
            (
                base_arrays.nfacet,
                base_arrays.nfacet,
                base_arrays.yB_size,
                base_arrays.yB_size,
            ),
            dtype=complex,
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(
            range(base_arrays.nsubgrid), range(base_arrays.nsubgrid)
        ):
            subgrid[i0][i1] = _ith_subgrid_facet_element(
                G,
                (
                    -base_arrays.subgrid_off[i0],
                    -base_arrays.subgrid_off[i1],
                ),
                base_arrays.xA_size,
                numpy.outer(
                    base_arrays.subgrid_A[i0],
                    base_arrays.subgrid_A[i1],
                ),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(
            range(base_arrays.nfacet), range(base_arrays.nfacet)
        ):
            facet[j0][j1] = _ith_subgrid_facet_element(
                FG,
                (
                    -base_arrays.facet_off[j0],
                    -base_arrays.facet_off[j1],
                ),
                base_arrays.yB_size,
                numpy.outer(base_arrays.facet_B[j0], base_arrays.facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet
