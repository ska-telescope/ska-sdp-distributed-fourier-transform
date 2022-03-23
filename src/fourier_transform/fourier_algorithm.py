"""Distributed Fourier Transform Module.
Included are a list of base functions that are used across the code.

"""
import itertools

import scipy.special
import scipy.signal
import numpy

from src.fourier_transform.dask_wrapper import dask_wrapper


def create_slice(fill_val, axis_val, dims, axis):
    """
    TODO: docstring + tests
    Slice A

    param fill_val: Fill value
    param axis_val: Axis value
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return tuple([axis_val if i == axis else fill_val for i in range(dims)])


def broadcast(a, dims, axis):
    """
    TODO: docstring + tests
    Broadcast A

    param a: A
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return a[create_slice(numpy.newaxis, slice(None), dims, axis)]


def pad_mid(a, n, axis):
    """
    Pad an array to a desired size with zeros at a given axis.
    (Surround the middle with zeros till it reaches the given size)

    param a: numpy array to be padded
    param n: size to be padded to (desired size)
    param axis: axis along which to pad

    return: padded numpy array
    """
    n0 = a.shape[axis]
    if n == n0:
        return a
    pad = create_slice(
        (0, 0), (n // 2 - n0 // 2, (n + 1) // 2 - (n0 + 1) // 2), len(a.shape), axis
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
    TODO: docstring + tests
    1D array which spans [-.5,.5[ with 0 at position N/2"""
    n2 = n // 2
    if n % 2 == 0:
        return numpy.mgrid[-n2:n2] / n
    else:
        return numpy.mgrid[-n2 : n2 + 1] / n


def anti_aliasing_function(shape, m, c):
    """
    TODO: tests
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair or just width
    :param m: mode parameter
    :param c: spheroidal parameter
    """

    # One dimensional?
    if len(numpy.array(shape).shape) == 0:

        pswf = scipy.special.pro_ang1(m, m, c, 2 * coordinates(shape))[0]
        pswf[0] = 0  # zap NaN
        return pswf

    # 2D Prolate spheroidal angular function is separable
    return numpy.outer(
        anti_aliasing_function(shape[0], m, c), anti_aliasing_function(shape[1], m, c)
    )


@dask_wrapper
def _ith_subgrid_facet_element(
    true_image, offset_i, true_usable_size, mask_element, axis=None, **kwargs
):
    """
    :param true_image: true image, G (1D or 2D)
    :param offset_i: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet
    :param mask_element: an element of subgrid_A or facet_B (masks)
    :param axis: axis (0 or 1)
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
                numpy.roll(true_image, offset_i, axis), true_usable_size, axis[0]
            ),
            true_usable_size,
            axis[1],
        )

    result = mask_element * extracted
    return result


def make_subgrid_and_facet(
    G,
    FG,
    constants_class,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
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
                (-constants_class.subgrid_off[i0], -constants_class.subgrid_off[i1]),
                constants_class.xA_size,
                numpy.outer(
                    constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
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
                (-constants_class.facet_off[j0], -constants_class.facet_off[j1]),
                constants_class.yB_size,
                numpy.outer(constants_class.facet_B[j0], constants_class.facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


# 2D FOURIER ALGORITHM FUNCTIONS (facet to subgrid)
@dask_wrapper
def prepare_facet(facet, axis, Fb, yP_size, **kwargs):
    """

    :param facet: Facet
    :param axis: Axis
    :param Fb:
    :param yP_size: Facet size, padded for m convolution (internal)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: BF
    """
    BF = pad_mid(facet * broadcast(Fb, len(facet.shape), axis), yP_size, axis)
    BF = ifft(BF, axis)
    return BF


@dask_wrapper
def extract_subgrid(
    BF,
    axis,
    subgrid_off_i,
    constants_class,
    **kwargs,
):
    """
    Extract the facet contribution of a subgrid.
    TODO: maybe rename function to reflect this definition
        See discussion at https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/merge_requests/4#note_825003275

    :param BF:
    :param axis: Axis
    :param subgrid_off_i:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return:
    """
    dims = len(BF.shape)
    BF_mid = extract_mid(
        numpy.roll(
            BF, -subgrid_off_i * constants_class.yP_size // constants_class.N, axis
        ),
        constants_class.xMxN_yP_size,
        axis,
    )
    MBF = broadcast(constants_class.facet_m0_trunc, dims, axis) * BF_mid
    MBF_sum = numpy.array(extract_mid(MBF, constants_class.xM_yP_size, axis))
    xN_yP_size = constants_class.xMxN_yP_size - constants_class.xM_yP_size
    # [:xN_yP_size//2] / [-xN_yP_size//2:] for axis, [:] otherwise
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    MBF_sum[slc1] += MBF[slc2]
    MBF_sum[slc2] += MBF[slc1]

    return broadcast(constants_class.Fn, len(BF.shape), axis) * extract_mid(
        fft(MBF_sum, axis), constants_class.xM_yN_size, axis
    )


# 2D FOURIER ALGORITHM FUNCTIONS (subgrid to facet)
@dask_wrapper
def prepare_subgrid(subgrid, xM_size, **kwargs):
    """
    Initial shared work per subgrid - no reason to do this per-axis, so always do it for all
    :param subgrid: Subgrid
    :param xM_size: Subgrid size, padded for transfer (internal)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: the FS term
    """
    padded = pad_mid(pad_mid(subgrid, xM_size, axis=0), xM_size, axis=1)
    ftransformed = fft(fft(padded, axis=0), axis=1)

    return ftransformed


@dask_wrapper
def extract_facet_contribution(FSi, facet_off_j, constants_class, axis, **kwargs):
    """
    Extract contribution of subgrid to a facet

    :param Fsi:
    :param facet_off_j:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param axis: Axis
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: Contribution of facet on the subgrid

    """
    return broadcast(constants_class.Fn, len(FSi.shape), axis) * extract_mid(
        numpy.roll(
            FSi, -facet_off_j * constants_class.xM_size // constants_class.N, axis
        ),
        constants_class.xM_yN_size,
        axis,
    )


@dask_wrapper
def add_subgrid_contribution(
    dims,
    NjSi,
    subgrid_off_i,
    constants_class,
    axis,
    **kwargs,
):
    """
    Add subgrid contribution to a facet

    :param MiNjSi_sum:
    :param NjSi:
    :param subgrid_off:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param axis:
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return MiNjSi_sum:

    """
    xN_yP_size = constants_class.xMxN_yP_size - constants_class.xM_yP_size
    NjSi_mid = ifft(pad_mid(NjSi, constants_class.xM_yP_size, axis), axis)
    NjSi_temp = pad_mid(NjSi_mid, constants_class.xMxN_yP_size, axis)
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    NjSi_temp[slc1] = NjSi_mid[slc2]
    NjSi_temp[slc2] = NjSi_mid[slc1]
    NjSi_temp = NjSi_temp * broadcast(
        constants_class.facet_m0_trunc, len(NjSi.shape), axis
    )

    return numpy.roll(
        pad_mid(NjSi_temp, constants_class.yP_size, axis),
        subgrid_off_i * constants_class.yP_size // constants_class.N,
        axis=axis,
    )


@dask_wrapper
def finish_facet(MiNjSi_sum, Fb, facet_B_j, yB_size, axis, **kwargs):
    """
    Obtain finished facet

    :param MiNjSi_sum:
    :param Fb:
    :param facet_B_j:
    :param yB_size: effective facet size
    :param axis:
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: The finished facet (in BMNAF term)
    """
    return extract_mid(fft(MiNjSi_sum, axis), yB_size, axis) * broadcast(
        Fb * facet_B_j, len(MiNjSi_sum.shape), axis
    )
