# pylint: disable=too-many-locals, too-many-arguments
# pylint: disable=unused-argument
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
def make_subgrid_and_facet(G, FG, constants_class, dims, use_dask=False):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param constants_class: BaseArrays or SparseFourierTransform class object
                            containing fundamental and derived parameters
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty(
            (constants_class.nsubgrid, constants_class.xA_size), dtype=complex
        )
        facet = numpy.empty(
            (constants_class.nfacet, constants_class.yB_size), dtype=complex
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(constants_class.nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G,
                -constants_class.subgrid_off[i],
                constants_class.xA_size,
                constants_class.subgrid_A[i],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(constants_class.nfacet):
            facet[j] = _ith_subgrid_facet_element(
                FG,
                -constants_class.facet_off[j],
                constants_class.yB_size,
                constants_class.facet_B[j],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty(
            (
                constants_class.nsubgrid,
                constants_class.nsubgrid,
                constants_class.xA_size,
                constants_class.xA_size,
            ),
            dtype=complex,
        )
        facet = numpy.empty(
            (
                constants_class.nfacet,
                constants_class.nfacet,
                constants_class.yB_size,
                constants_class.yB_size,
            ),
            dtype=complex,
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(
            range(constants_class.nsubgrid), range(constants_class.nsubgrid)
        ):
            subgrid[i0][i1] = _ith_subgrid_facet_element(
                G,
                (
                    -constants_class.subgrid_off[i0],
                    -constants_class.subgrid_off[i1],
                ),
                constants_class.xA_size,
                numpy.outer(
                    constants_class.subgrid_A[i0],
                    constants_class.subgrid_A[i1],
                ),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(
            range(constants_class.nfacet), range(constants_class.nfacet)
        ):
            facet[j0][j1] = _ith_subgrid_facet_element(
                FG,
                (
                    -constants_class.facet_off[j0],
                    -constants_class.facet_off[j1],
                ),
                constants_class.yB_size,
                numpy.outer(
                    constants_class.facet_B[j0], constants_class.facet_B[j1]
                ),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


# ----------pure---function------------
def prepare_facet(facet, axis, Fb, yP_size, **kwargs):
    """
    Calculate the inverse FFT of a padded facet element multiplied by Fb
    (Fb: Fourier transform of grid correction function)

    :param facet: single facet element
    :param axis: axis along which operations are performed (0 or 1)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: TODO: BF? prepared facet
    """
    BF = pad_mid(facet * broadcast(Fb, len(facet.shape), axis), yP_size, axis)
    BF = ifft(BF, axis)
    return BF


def extract_facet_contrib_to_subgrid(
    BF,
    axis,
    subgrid_off_elem,
    yP_size,
    xMxN_yP_size,
    xM_yP_size,
    xM_yN_size,
    N,
    Fn,
    facet_m0_trunc,
    **kwargs
):
    """
    Extract the facet contribution to a subgrid.

    :param BF: TODO: ? prepared facet
    :param axis: axis along which the operations are performed (0 or 1)
    :param subgrid_off_elem: single subgrid offset element
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: contribution of facet to subgrid
    """
    dims = len(BF.shape)
    BF_mid = extract_mid(
        numpy.roll(BF, -subgrid_off_elem * yP_size // N, axis),
        xMxN_yP_size,
        axis,
    )
    MBF = broadcast(facet_m0_trunc, dims, axis) * BF_mid
    MBF_sum = numpy.array(extract_mid(MBF, xM_yP_size, axis))
    xN_yP_size = xMxN_yP_size - xM_yP_size
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    MBF_sum[slc1] += MBF[slc2]
    MBF_sum[slc2] += MBF[slc1]

    return broadcast(Fn, len(BF.shape), axis) * extract_mid(
        fft(MBF_sum, axis), xM_yN_size, axis
    )


def add_facet_contribution(
    facet_contrib, facet_off_elem, axis, xM_size, N, **kwargs
):
    """
    Further transforms facet contributions, which then will be summed up.

    :param facet_contrib: array-chunk of individual facet contributions
    :param facet_off_elem: facet offset for the facet_contrib array chunk
    :param axis: axis along which the operations are performed (0 or 1)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: TODO??
    """
    return numpy.roll(
        pad_mid(facet_contrib, xM_size, axis),
        facet_off_elem * xM_size // N,
        axis=axis,
    )


def finish_subgrid(
    summed_facets, subgrid_mask1, subgrid_mask2, xA_size, **kwargs
):
    """
    Obtain finished subgrid.
    Operation performed for both axis (only works on 2D arrays).

    :param summed_facets: summed facets contributing to thins subgrid
    :param subgrid_mask1: ith subgrid mask element
    :param subgrid_mask2: (i+1)th subgrid mask element
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: approximate subgrid element
    """
    tmp = extract_mid(
        extract_mid(
            ifft(ifft(summed_facets, axis=0), axis=1), xA_size, axis=0
        ),
        xA_size,
        axis=1,
    )
    approx_subgrid = tmp * numpy.outer(subgrid_mask1, subgrid_mask2)
    return approx_subgrid


def prepare_subgrid(subgrid, xM_size, **kwargs):
    """
    Calculate the FFT of a padded subgrid element.
    No reason to do this per-axis, so always do it for both axis.
    (Note: it will only work for 2D subgrid arrays)

    :param subgrid: single subgrid array element
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: TODO: the FS ??? term
    """
    padded = pad_mid(pad_mid(subgrid, xM_size, axis=0), xM_size, axis=1)
    fftd = fft(fft(padded, axis=0), axis=1)

    return fftd


def extract_subgrid_contrib_to_facet(
    FSi, facet_off_elem, axis, xM_size, xM_yN_size, N, Fn, **kwargs
):
    """
    Extract contribution of subgrid to a facet.

    :param Fsi: TODO???
    :param facet_off_elem: single facet offset element
    :param axis: axis along which the operations are performed (0 or 1)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: Contribution of subgrid to facet

    """
    return broadcast(Fn, len(FSi.shape), axis) * extract_mid(
        numpy.roll(FSi, -facet_off_elem * xM_size // N, axis), xM_yN_size, axis
    )


def add_subgrid_contribution(
    dims,
    NjSi,
    subgrid_off_elem,
    axis,
    xMxN_yP_size,
    xM_yP_size,
    yP_size,
    N,
    facet_m0_trunc,
    **kwargs
):
    """
    Further transform subgrid contributions, which are then summed up.

    :param dims: length of tuple to be produced by create_slice
                 (i.e. number of dimensions); int
    :param NjSi: TODO
    :param subgrid_off_elem: single subgrid offset element
    :param axis: axis along which operations are performed (0 or 1)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return summed subgrid contributions

    """
    xN_yP_size = xMxN_yP_size - xM_yP_size
    NjSi_mid = ifft(pad_mid(NjSi, xM_yP_size, axis), axis)
    NjSi_temp = pad_mid(NjSi_mid, xMxN_yP_size, axis)
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    NjSi_temp[slc1] = NjSi_mid[slc2]
    NjSi_temp[slc2] = NjSi_mid[slc1]
    NjSi_temp = NjSi_temp * broadcast(facet_m0_trunc, len(NjSi.shape), axis)

    return numpy.roll(
        pad_mid(NjSi_temp, yP_size, axis),
        subgrid_off_elem * yP_size // N,
        axis=axis,
    )


def finish_facet(MiNjSi_sum, facet_B_elem, axis, yB_size, Fb, **kwargs):
    """
    Obtain finished facet.

    It extracts from the padded facet (obtained from subgrid via FFT)
    the true-sized facet and multiplies with masked Fb.
    (Fb: Fourier transform of grid correction function)

    :param MiNjSi_sum: sum of subgrid contributions to a facet
    :param facet_B_elem: a facet mask element
    :param axis: axis along which operations are performed (0 or 1)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: finished (approximate) facet element
    """
    return extract_mid(fft(MiNjSi_sum, axis), yB_size, axis) * broadcast(
        Fb * facet_B_elem, len(MiNjSi_sum.shape), axis
    )
