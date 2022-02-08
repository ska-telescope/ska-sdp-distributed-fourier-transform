"""Distributed Fourier Transform Module."""
import scipy.special
import scipy.signal
import numpy
import dask
import dask.array
from src.fourier_transform.dask_wrapper import dask_wrapper

# TODO: ideally we'd like to merge the 1D functions with their 2D equivalent,
#   which then can be used for both versions


# TODO: need to merge the functions with _a in their name to their equivalents from Crocodile
#   Crocodile functions are below the _a ones
def slice_a(fill_val, axis_val, dims, axis):
    """
    Slice A

    param fill_val: Fill value
    param axis_val: Axis value
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return tuple([axis_val if i == axis else fill_val for i in range(dims)])


def pad_mid_a(a, N, axis):
    """
    Pad Mid A

    param a: A
    param N: Total image size on a side
    param axis: Axis

    return:
    """
    N0 = a.shape[axis]
    if N == N0:
        return a
    pad = slice_a(
        (0, 0), (N // 2 - N0 // 2, (N + 1) // 2 - (N0 + 1) // 2), len(a.shape), axis
    )
    return numpy.pad(a, pad, mode="constant", constant_values=0.0)


def extract_mid_a(a, N, axis):
    """
    Extract mid A

    param a: A
    param N: Total image size on a side
    param axis: Axis

    return:
    """
    assert N <= a.shape[axis]
    cx = a.shape[axis] // 2
    if N % 2 != 0:
        slc = slice(cx - N // 2, cx + N // 2 + 1)
    else:
        slc = slice(cx - N // 2, cx + N // 2)
    return a[slice_a(slice(None), slc, len(a.shape), axis)]


def fft_a(a, axis):
    """
    FFT A

    param a: A
    param axis: Axis

    return:
    """
    return numpy.fft.fftshift(
        numpy.fft.fft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def ifft_a(a, axis):
    """
    IFFT A

    param a: A
    param axis: Axis

    return:
    """
    return numpy.fft.fftshift(
        numpy.fft.ifft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def broadcast_a(a, dims, axis):
    """
    Broadcast A

    param a: A
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return a[slice_a(numpy.newaxis, slice(None), dims, axis)]


# CROCODILE FUNCTIONS:
# TODO: update, where needed, so they can also operate a single, given axis (merge with funcs above)
def coordinates(N):
    """1D array which spans [-.5,.5[ with 0 at position N/2"""
    N2 = N // 2
    if N % 2 == 0:
        return numpy.mgrid[-N2:N2] / N
    else:
        return numpy.mgrid[-N2 : N2 + 1] / N


def fft(a):
    """Fourier transformation from image to grid space

    :param a: image in `lm` coordinate space
    :param axis: int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
        (doc from numpy.fft.ifftshift docstring)
    :return: `uv` grid
    """
    if len(a.shape) == 1:
        return numpy.fft.fftshift(numpy.fft.fft(numpy.fft.ifftshift(a)))
    elif len(a.shape) == 2:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))
    assert False, "Unsupported image shape for FFT!"


def ifft(a):
    """Fourier transformation from grid to image space

    :param a: `uv` grid to transform
    :return: an image in `lm` coordinate space
    """
    if len(a.shape) == 1:
        return numpy.fft.fftshift(numpy.fft.ifft(numpy.fft.ifftshift(a)))
    elif len(a.shape) == 2:
        return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))
    assert False, "Unsupported grid shape for iFFT!"


def pad_mid(ff, N):
    """
    Pad a far field image with zeroes to make it the given size.

    Effectively as if we were multiplying with a box function of the
    original field's size, which is equivalent to a convolution with a
    sinc pattern in the uv-grid.

    :param ff: The input far field. Should be smaller than NxN.
    :param N:  The desired far field size

    """

    N0 = ff.shape[0]
    if N == N0:
        return ff
    assert N > N0
    pad = [(N // 2 - N0 // 2, (N + 1) // 2 - (N0 + 1) // 2)]
    if len(ff.shape) == 2:
        assert N0 == ff.shape[1]
        pad = 2 * pad  # both dimensions
    return numpy.pad(ff, pad, mode="constant", constant_values=0.0)


def extract_mid(a, N):
    """
    Extract a section from middle of a map

    Suitable for zero frequencies at N/2. This is the reverse
    operation to pad.

    :param a: grid from which to extract
    :param s: size of section
    """

    assert N <= a.shape[0]
    cx = a.shape[0] // 2
    s = N // 2
    if len(a.shape) == 2:
        assert N <= a.shape[1]
        cy = a.shape[1] // 2
        if N % 2 != 0:
            return a[cx - s : cx + s + 1, cy - s : cy + s + 1]
        else:
            return a[cx - s : cx + s, cy - s : cy + s]
    elif len(a.shape) == 1:
        if N % 2 != 0:
            return a[cx - s : cx + s + 1]
        else:
            return a[cx - s : cx + s]
    else:
        assert False, "Unsupported grid shape for extract_mid!"


def anti_aliasing_function(shape, m, c):
    """
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


# 1D FOURIER ALGORITHM FUNCTIONS
@dask_wrapper
def _ith_subgrid_facet_element(
    true_image, offset_i, true_usable_size, mask_element, axis=None, **kwargs
):
    result = mask_element * extract_mid(
        numpy.roll(true_image, offset_i, axis), true_usable_size
    )
    return result


def make_subgrid_and_facet(
    G,
    FG,
    nsubgrid,
    xA_size,
    subgrid_A,
    subgrid_off,
    nfacet,
    yB_size,
    facet_B,
    facet_off,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param nsubgrid: number of subgrid
    :param xA_size: true usable subgrid size
    :param subgrid_A: subgrid mask
    :param subgrid_off: subgrid offset
    :param nfacet: number of facet
    :param yB_size: effective facet size
    :param facet_B: facet mask
    :param facet_off: facet offset
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty((nsubgrid, xA_size), dtype=complex)
        facet = numpy.empty((nfacet, yB_size), dtype=complex)

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G,
                -subgrid_off[i],
                xA_size,
                subgrid_A[i],
                axis=None,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(nfacet):
            facet[j] = _ith_subgrid_facet_element(
                FG,
                -facet_off[j],
                yB_size,
                facet_B[j],
                axis=None,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty((nsubgrid, nsubgrid, xA_size, xA_size), dtype=complex)
        facet = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
            subgrid[i0][i1] = _ith_subgrid_facet_element(
                G,
                (-subgrid_off[i0], -subgrid_off[i1]),
                xA_size,
                numpy.outer(subgrid_A[i0], subgrid_A[i1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
            facet[j0][j1] = _ith_subgrid_facet_element(
                FG,
                (-facet_off[j0], -facet_off[j1]),
                yB_size,
                numpy.outer(facet_B[j0], facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


def make_subgrid_and_facet_dask_array(
    G, nsubgrid, xA_size, subgrid_A, subgrid_off, nfacet, yB_size, facet_B, facet_off,
):
    """
    Calculate the actual subgrids and facets. Same as make_subgrid_and_facet
    but implemented using dask.array.

    :param G: "ground truth", the actual input data
    :param nsubgrid: number of subgrid
    :param xA_size: true usable subgrid size
    :param subgrid_A: subgrid mask
    :param subgrid_off: subgrid offset
    :param nfacet: number of facet
    :param yB_size: effective facet size
    :param facet_B: facet mask
    :param facet_off: facet offset

    :return: tuple of two dask.array (subgrid, facet)
    """

    # TODO: I didn't change the FG variable and am currently not using use_dask=True.
    # TODO: But this may need to be changed
    FG = fft(G)

    subgrid = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                G, -subgrid_off[i], xA_size, subgrid_A[i], axis=None
            )
            for i in range(nsubgrid)
        ],
        chunks=(1, xA_size),
    ).astype(complex)

    facet = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                FG, -facet_off[j], yB_size, facet_B[j], axis=None
            )
            for j in range(nfacet)
        ],
        chunks=(1, yB_size),
    ).astype(complex)

    return subgrid, facet


@dask_wrapper
def facet_contribution_to_subgrid_1d(
    BjFj,
    facet_m0_trunc,
    offset_i,
    yP_size,
    N,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    xM_yN_size,
    Fn,
    **kwargs
):
    """
    Extract the facet contribution to a subgrid for 1D version.

    :param BjFj: Prepared facet data (i.e. multiplied by b, padded to yP_size, Fourier transformed)
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param offset_i: ith offset value (subgrid)
    :param yP_size: facet size, padded for m convolution (internal; it's len(BjFj))
    :param N: total image size on a side
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param Fn: Fourier transform of gridding function

    :return: facet_in_a_subgrid: facet contribution to a subgrid
    """
    MiBjFj = facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * yP_size // N), xMxN_yP_size
    )
    MiBjFj_sum = extract_mid(MiBjFj, xM_yP_size)
    MiBjFj_sum[: xN_yP_size // 2] = (
        MiBjFj_sum[: xN_yP_size // 2] + MiBjFj[-xN_yP_size // 2 :]
    )
    MiBjFj_sum[-xN_yP_size // 2 :] = (
        MiBjFj_sum[-xN_yP_size // 2 :] + MiBjFj[: xN_yP_size // 2 :]
    )

    facet_in_a_subgrid = Fn * extract_mid(fft(MiBjFj_sum), xM_yN_size)

    return facet_in_a_subgrid


def facet_contribution_to_subgrid_1d_dask_array(
    BjFj,
    facet_m0_trunc,
    offset_i,
    yP_size,
    N,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    xM_yN_size,
    Fn,
):
    """
    Extract the facet contribution of a subgrid for 1D version.
    Same as facet_contribution_to_subgrid_1d but implemented using dask.array.

    :param BjFj: Prepared facet data (i.e. multiplied by b, padded to yP_size, Fourier transformed)
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param offset_i: ith offset value (subgrid)
    :param yP_size: facet size, padded for m convolution (internal; it's len(BjFj))
    :param N: total image size on a side
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param Fn: Fourier transform of gridding function

    :return: facet_in_a_subgrid: facet contribution to a subgrid
    """
    MiBjFj = facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * yP_size // N), xMxN_yP_size
    ).rechunk(xMxN_yP_size)
    MiBjFj_sum = extract_mid(MiBjFj, xM_yP_size).rechunk(xM_yP_size)
    MiBjFj_sum[: xN_yP_size // 2] += MiBjFj[-xN_yP_size // 2 :]
    MiBjFj_sum[-xN_yP_size // 2 :] += MiBjFj[: xN_yP_size // 2 :]

    facet_in_a_subgrid = Fn * extract_mid(fft(MiBjFj_sum), xM_yN_size).rechunk(
        xM_yN_size
    )

    return facet_in_a_subgrid


@dask_wrapper
def prepare_facet_1d(facet_j, Fb, yP_size, **kwargs):

    return ifft(pad_mid(facet_j * Fb, yP_size))  # prepare facet


def facets_to_subgrid_1d(
    facet,
    nsubgrid,
    nfacet,
    xM_yN_size,
    Fb,
    Fn,
    yP_size,
    facet_m0_trunc,
    subgrid_off,
    N,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    dtype,
    use_dask,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.

    :param facet: numpy array of facets
    :param nsubgrid: number of subgrid
    :param nfacet: number of facet
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param Fb: Fourier transform of grid correction function
    :param Fn: Fourier transform of gridding function
    :param yP_size: padded (rough) image size (facet)
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param subgrid_off: subgrid offset
    :param N: total image size on a side
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param dtype: data type
    :param use_dask: use Dask?

    :return: RNjMiBjFj: array of contributions of this facet to different subgrids
    """
    RNjMiBjFj = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=dtype)

    if use_dask:
        RNjMiBjFj = RNjMiBjFj.tolist()

    for j in range(nfacet):
        BjFj = prepare_facet_1d(facet[j], Fb, yP_size, use_dask=True, nout=1)
        for i in range(nsubgrid):
            RNjMiBjFj[i][j] = facet_contribution_to_subgrid_1d(  # extract subgrid
                BjFj,
                facet_m0_trunc,
                subgrid_off[i],
                yP_size,
                N,
                xMxN_yP_size,
                xN_yP_size,
                xM_yP_size,
                xM_yN_size,
                Fn,
                use_dask=use_dask,
                nout=1,
            )

    return RNjMiBjFj


def facets_to_subgrid_1d_dask_array(
    facet,
    nsubgrid,
    nfacet,
    xM_yN_size,
    Fb,
    Fn,
    yP_size,
    facet_m0_trunc,
    subgrid_off,
    N,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    dtype,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.
    Same as facets_to_subgrid_1d but implemented using dask.array.

    :param facet: numpy array of facets
    :param nsubgrid: number of subgrid
    :param nfacet: number of facet
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param Fb: Fourier transform of grid correction function
    :param Fn: Fourier transform of gridding function
    :param yP_size: padded (rough) image size (facet)
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param subgrid_off: subgrid offset
    :param N: total image size on a side
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param dtype: data type

    :return: RNjMiBjFj: array of contributions of this facet to different subgrids
    """
    RNjMiBjFj = dask.array.from_array(
        [
            [
                facet_contribution_to_subgrid_1d_dask_array(
                    ifft(
                        pad_mid(facet[i] * Fb, yP_size).rechunk(yP_size)
                    ),  # prepare facet
                    facet_m0_trunc,
                    subgrid_off[j],
                    yP_size,
                    N,
                    xMxN_yP_size,
                    xN_yP_size,
                    xM_yP_size,
                    xM_yN_size,
                    Fn,
                )
                for i in range(nfacet)
            ]
            for j in range(nsubgrid)
        ],
        chunks=(1, 1, xM_yN_size),
    ).astype(dtype)
    return RNjMiBjFj


@dask_wrapper
def add_padded_value(nmbf, facet_off_j, xM_size, N, **kwargs):
    return numpy.roll(pad_mid(nmbf, xM_size), facet_off_j * xM_size // N)


def reconstruct_subgrid_1d(
    nmbfs, xM_size, nfacet, facet_off, N, subgrid_A, xA_size, nsubgrid, use_dask
):
    """
    Reconstruct the subgrid array calculated by facets_to_subgrid_1d.
    TODO: why are we doing this? how is the result different from the result of facets_to_subgrid_1d?

    Note: compared to reconstruct_facet_1d, which does the same but for facets,
    the order in which we perform calculations is different:
    Here we first do the sum, and then cut from xM_size to xA_size,
    hence approx starts as a larger sized array;
    In reconstruct_facet_1d we apply fft then extract_mid to cut down from size yP_size to yB_size,
    finally we do the sum (which results in approx), hence approx is already defined with the smaller size.

    :param nmbfs: subgrid array calculated from facets by facets_to_subgrid_1d
    :param xM_size: padded (rough) subgrid size
    :param nfacet: number of facet
    :param facet_off: facet offset
    :param N: total image size on a side
    :param subgrid_A: subgrid mask
    :param xA_size: true usable subgrid size
    :param use_dask: use Dask?

    :return: approximate subgrid
    """
    approx_subgrid = numpy.ndarray((nsubgrid, xA_size), dtype=complex)
    if use_dask:
        approx_subgrid = approx_subgrid.tolist()
        approx = numpy.zeros((nsubgrid, xM_size), dtype=complex)
        approx = approx.tolist()
        for i in range(nsubgrid):
            for j in range(nfacet):
                approx[i] = approx[i] + add_padded_value(
                    nmbfs[i][j], facet_off[j], xM_size, N, use_dask=use_dask, nout=1
                )
            # TODO: Here we used dask array in order to avoid complications of ifft, but this is not optimal.
            approx_array = dask.array.from_delayed(approx[i], (xM_size,), dtype=complex)
            approx_subgrid[i] = subgrid_A[i] * extract_mid(ifft(approx_array), xA_size)
    else:
        for i in range(nsubgrid):
            approx = numpy.zeros(xM_size, dtype=complex)
            for j in range(nfacet):
                approx = approx + add_padded_value(
                    nmbfs[i, j], facet_off[j], xM_size, N
                )
            approx_subgrid[i, :] = subgrid_A[i] * extract_mid(ifft(approx), xA_size)

    return approx_subgrid


def reconstruct_subgrid_1d_dask_array(
    nmbfs, xM_size, nfacet, facet_off, N, subgrid_A, xA_size, nsubgrid
):
    """
    Reconstruct the subgrid array calculated by facets_to_subgrid_1d.
    Same as reconstruct_subgrid_1d but using dask.array.
    TODO: same questions as for redistribute_subgrid_1d

    :param nmbfs: subgrid array calculated from facets by facets_to_subgrid_1d
    :param xM_size: padded (rough) subgrid size
    :param nfacet: number of facet
    :param facet_off: facet offset
    :param N: total image size on a side
    :param subgrid_A: subgrid mask
    :param xA_size: true usable subgrid size

    :return: approximate subgrid
    """
    approx = dask.array.from_array(
        numpy.zeros((nsubgrid, xM_size), dtype=complex), chunks=(1, xM_size)
    )
    approx_subgrid = dask.array.from_array(
        numpy.ndarray((nsubgrid, xA_size), dtype=complex), chunks=(1, xA_size)
    )
    for i in range(nsubgrid):
        for j in range(nfacet):
            padded = pad_mid(nmbfs[i, j, :], xM_size).rechunk(xM_size)
            approx[i, :] += numpy.roll(padded, facet_off[j] * xM_size // N)

        approx_subgrid[i, :] = subgrid_A[i] * extract_mid(
            ifft(approx[i]), xA_size
        ).rechunk(xA_size)

    return approx_subgrid


@dask_wrapper
def calculate_fns_term(
    subgrid_ith, facet_off_jth, Fn, xM_size, xM_yN_size, N, **kwargs
):

    return Fn * extract_mid(
        numpy.roll(fft(pad_mid(subgrid_ith, xM_size)), -facet_off_jth * xM_size // N),
        xM_yN_size,
    )


def subgrid_to_facet_1d(
    subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn, use_dask
):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.

    :param subgrid: numpy array of subgrids
    :param nsubgrid: number of subgrid
    :param nfacet: number of facets
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param xM_size: padded subgrid size
    :param facet_off: facet offset
    :param N: total image size on a side
    :param Fn: Fourier transform of gridding function
    :param use_dask: use Dask?

    :return: distributed facet array determined from input subgrid array
    """
    FNjSi = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=complex)

    if use_dask:
        FNjSi = FNjSi.tolist()

    for i in range(nsubgrid):
        for j in range(nfacet):
            FNjSi[i][j] = calculate_fns_term(
                subgrid[i],
                facet_off[j],
                Fn,
                xM_size,
                xM_yN_size,
                N,
                use_dask=use_dask,
                nout=1,
            )

    return FNjSi


def subgrid_to_facet_1d_dask_array(
    subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.
    Same as subgrid_to_facet_1d but implemented with dask.array

    :param subgrid: numpy array of subgrids
    :param nsubgrid: number of subgrid
    :param nfacet: number of facets
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param xM_size: padded subgrid size
    :param facet_off: facet offset
    :param N: total image size on a side
    :param Fn: Fourier transform of gridding function

    :return: distributed facet array determined from input subgrid array
    """
    FNjSi = dask.array.from_array(
        [
            [
                extract_mid(
                    numpy.roll(
                        fft(pad_mid(subgrid[j], xM_size).rechunk(xM_size)),
                        -facet_off[i] * xM_size // N,
                    ),
                    xM_yN_size,
                )
                for i in range(nfacet)
            ]
            for j in range(nsubgrid)
        ]
    )
    distributed_facet = Fn * FNjSi

    return distributed_facet


@dask_wrapper
def add_subgrid_contribution_1d(
    xMxN_yP_size,
    xM_yP_size,
    nafs_ij,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off_i,
    yB_size,
    N,
    **kwargs
):
    """
    Add subgrid contribution to a single facet.

    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param nafs_ij: redistributed facet array [i:j] element
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param yP_size: facet size padded for m convolution (internal)
    :param subgrid_off_i: subgrid offset [i]th element
    :param yB_size: effective facet size
    :param N: total image size on a side

    :return: subgrid contribution
    """
    NjSi = numpy.zeros(xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, xM_yP_size)
    NjSi_mid[:] = ifft(pad_mid(nafs_ij, xM_yP_size))  # updates NjSi via reference!
    NjSi[-xN_yP_size // 2 :] = NjSi_mid[: xN_yP_size // 2]
    NjSi[: xN_yP_size // 2 :] = NjSi_mid[-xN_yP_size // 2 :]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(facet_m0_trunc * NjSi, yP_size), subgrid_off_i * yP_size // N
        )
    )
    subgrid_contrib = extract_mid(FMiNjSi, yB_size)
    return subgrid_contrib


def add_subgrid_contribution_1d_dask_array(
    xMxN_yP_size,
    xM_yP_size,
    nafs_ij,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off_i,
    yB_size,
    N,
):
    """
    Add subgrid contribution to a single facet.
    Same as add_subgrid_contribution_1d but implemented with dask.array.

    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param nafs_ij: redistributed facet array [i:j] element
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param yP_size: facet size padded for m convolution (internal)
    :param subgrid_off_i: subgrid offset [i]th element
    :param yB_size: effective facet size
    :param N: total image size on a side

    :return: subgrid contribution
    """
    NjSi = numpy.zeros(xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, xM_yP_size)
    NjSi_mid[:] = ifft(
        pad_mid(nafs_ij, xM_yP_size).rechunk(xM_yP_size)
    )  # updates NjSi via reference!
    NjSi[-xN_yP_size // 2 :] = NjSi_mid[: xN_yP_size // 2]
    NjSi[: xN_yP_size // 2 :] = NjSi_mid[-xN_yP_size // 2 :]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(facet_m0_trunc * NjSi, yP_size), subgrid_off_i * yP_size // N
        )
    )
    subgrid_contrib = extract_mid(FMiNjSi, yB_size)
    return subgrid_contrib


def reconstruct_facet_1d(
    nafs,
    nfacet,
    yB_size,
    nsubgrid,
    xMxN_yP_size,
    xM_yP_size,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off,
    N,
    Fb,
    facet_B,
    use_dask=False,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d

    :param nafs: redistributed facet array calculated from facets by subgrid_to_facet_1d
    :param nfacet: number of facets
    :param yB_size: effective facet size
    :param nsubgrid: number of subgrids
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param yP_size: facet size padded for m convolution (internal)
    :param subgrid_off: subgrid offset
    :param N: total image size on a side
    :param Fb: Fourier transform of grid correction function
    :param facet_B: facet mask

    :return: approximate facet
    """
    approx_facet = numpy.ndarray((nfacet, yB_size), dtype=complex)
    if use_dask:
        approx_facet = approx_facet.tolist()
        approx = numpy.zeros((nfacet, yB_size), dtype=complex)
        approx = approx.tolist()
        for j in range(nfacet):
            for i in range(nsubgrid):
                approx[j] = approx[j] + add_subgrid_contribution_1d(
                    xMxN_yP_size,
                    xM_yP_size,
                    nafs[i][j],
                    xN_yP_size,
                    facet_m0_trunc,
                    yP_size,
                    subgrid_off[i],
                    yB_size,
                    N,
                    use_dask=use_dask,
                    nout=1,
                )
            approx_facet[j] = approx[j] * Fb * facet_B[j]
    else:
        for j in range(nfacet):
            approx = numpy.zeros(yB_size, dtype=complex)
            for i in range(nsubgrid):
                approx = approx + add_subgrid_contribution_1d(
                    xMxN_yP_size,
                    xM_yP_size,
                    nafs[i, j],
                    xN_yP_size,
                    facet_m0_trunc,
                    yP_size,
                    subgrid_off[i],
                    yB_size,
                    N,
                )
            approx_facet[j, :] = approx * Fb * facet_B[j]

    return approx_facet


def reconstruct_facet_1d_dask_array(
    nafs,
    nfacet,
    yB_size,
    nsubgrid,
    xMxN_yP_size,
    xM_yP_size,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off,
    N,
    Fb,
    facet_B,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d.
    Same as reconstruct_facet_1d but implemented with dask.array.

    :param nafs: redistributed facet array calculated from facets by subgrid_to_facet_1d
    :param nfacet: number of facets
    :param yB_size: effective facet size
    :param nsubgrid: number of subgrids
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xM_yP_size: (padded subgrid size * padded image size (facet)) / N
    :param xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracetd of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param yP_size: facet size padded for m convolution (internal)
    :param subgrid_off: subgrid offset
    :param N: total image size on a side
    :param Fb: Fourier transform of grid correction function
    :param facet_B: facet mask

    :return: approximate facet
    """
    approx = dask.array.from_array(
        numpy.zeros((nfacet, yB_size), dtype=complex), chunks=(1, yB_size)
    )  # why is this not yP_size? for the subgrid version it uses xM_size
    approx_facet = dask.array.from_array(
        numpy.ndarray((nfacet, yB_size), dtype=complex), chunks=(1, yB_size)
    )
    for j in range(nfacet):
        for i in range(nsubgrid):
            approx[j, :] += add_subgrid_contribution_1d_dask_array(
                xMxN_yP_size,
                xM_yP_size,
                nafs[i, j],
                xN_yP_size,
                facet_m0_trunc,
                yP_size,
                subgrid_off[i],
                yB_size,
                N,
            )
        approx_facet[j, :] = approx[j] * Fb * facet_B[j]

    return approx_facet


# 2D FOURIER ALGORITHM FUNCTIONS (facet to subgrid)
def prepare_facet(facet, axis, Fb, yP_size):
    """

    param facet: Facet
    param axis: Axis
    param Fb:
    param yP_size: Facet size, padded for m convolution (internal)


    return: BF
    """
    BF = pad_mid_a(facet * broadcast_a(Fb, len(facet.shape), axis), yP_size, axis)
    BF = ifft_a(BF, axis)
    return BF


def extract_subgrid(
    BF,
    i,
    axis,
    subgrid_off,
    yP_size,
    xMxN_yP_size,
    facet_m0_trunc,
    xM_yP_size,
    Fn,
    xM_yN_size,
    N,
):
    """
    Extract the facet contribution of a subgrid.
    TODO: maybe rename function to reflect this definition
        See discussion at https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/merge_requests/4#note_825003275

    :param BF:
    :param i:
    :param axis: Axis
    :param subgrid_off:
    :param yP_size: Facet size, padded for m convolution (internal)
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param facet_m0_trunc:
    :param xM_yP_size:
    :param Fn:
    :param xM_yN_size:
    :param N: Total image size on a side

    :return:
    """
    dims = len(BF.shape)
    BF_mid = extract_mid_a(
        numpy.roll(BF, -subgrid_off[i] * yP_size // N, axis), xMxN_yP_size, axis
    )
    MBF = broadcast_a(facet_m0_trunc, dims, axis) * BF_mid
    MBF_sum = numpy.array(extract_mid_a(MBF, xM_yP_size, axis))
    xN_yP_size = xMxN_yP_size - xM_yP_size
    # [:xN_yP_size//2] / [-xN_yP_size//2:] for axis, [:] otherwise
    slc1 = slice_a(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = slice_a(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    MBF_sum[slc1] += MBF[slc2]
    MBF_sum[slc2] += MBF[slc1]

    return broadcast_a(Fn, len(BF.shape), axis) * extract_mid_a(
        fft_a(MBF_sum, axis), xM_yN_size, axis
    )


# 2D FOURIER ALGORITHM FUNCTIONS (subgrid to facet)
def prepare_subgrid(subgrid, xM_size):
    """
    Initial shared work per subgrid - no reason to do this per-axis, so always do it for all
    :param subgrid: Subgrid
    :param xM_size: Subgrid size, padded for transfer (internal)

    :return: the FS term
    """
    return fft(pad_mid(subgrid, xM_size))


def extract_facet_contribution(FSi, Fn, facet_off, j, xM_size, N, xM_yN_size, axis):
    """
    Extract contribution of subgrid to a facet

    :param Fsi:
    :param Fn:
    :param facet_off:
    :param j: Index on the facet
    :param xM_size: Subgrid size, padded for transfer (internal)
    :param N: Total image size on a side
    :param xM_yN_size:
    :param axis: Axis

    :return: Contribution of facet on the subgrid

    """

    return broadcast_a(Fn, len(FSi.shape), axis) * extract_mid_a(
        numpy.roll(FSi, -facet_off[j] * xM_size // N, axis), xM_yN_size, axis
    )


def add_subgrid_contribution(
    MiNjSi_sum,
    NjSi,
    i,
    facet_m0_trunc,
    subgrid_off,
    xMxN_yP_size,
    xM_yP_size,
    yP_size,
    N,
    axis,
):
    """
    Add subgrid contribution to a facet

    :param MiNjSi_sum:
    :param NjSi:
    :param i:
    :param facet_m0_trunc:
    :param subgrid_off:
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param xM_yP_size:
    :param yP_size:
    :param N:
    :param axis:

    :return MiNjSi_sum:

    """
    dims = len(MiNjSi_sum.shape)
    MiNjSi = numpy.zeros_like(MiNjSi_sum)
    NjSi_temp = extract_mid_a(MiNjSi, xMxN_yP_size, axis)
    NjSi_mid = extract_mid_a(NjSi_temp, xM_yP_size, axis)
    NjSi_mid[...] = ifft_a(
        pad_mid_a(NjSi, xM_yP_size, axis), axis
    )  # updates NjSi via reference!
    xN_yP_size = xMxN_yP_size - xM_yP_size
    slc1 = slice_a(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = slice_a(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    NjSi_temp[slc1] = NjSi_mid[slc2]
    NjSi_temp[slc2] = NjSi_mid[slc1]
    NjSi_temp[...] *= broadcast_a(facet_m0_trunc, len(NjSi.shape), axis)
    MiNjSi_sum[...] += numpy.roll(
        pad_mid_a(MiNjSi, yP_size, axis), subgrid_off[i] * yP_size // N, axis=axis
    )


def finish_facet(MiNjSi_sum, Fb, facet_B, yB_size, j, axis):
    """
    Obtain finished facet

    :param MiNjSi_sum:
    :param Fb:
    :param facet_B:
    :param yB_size: effective facet size
    :param j:
    :param axis:

    :return: The finished facet (in BMNAF term)
    """
    return extract_mid_a(fft_a(MiNjSi_sum, axis), yB_size, axis) * broadcast_a(
        Fb * facet_B[j], len(MiNjSi_sum.shape), axis
    )


# COMMON 1D and 2D FUNCTIONS -- SETTING UP FOR ALGORITHM TO RUN
def calculate_pswf(yN_size, alpha, W):
    """
    Calculate PSWF (prolate-spheroidal wave function) at the
    full required resolution (facet size)

    :param yN_size: needed padding
    :param alpha: TODO: ???, int
    :param W: PSWF parameter (grid-space support), float
    """
    pswf = anti_aliasing_function(yN_size, alpha, numpy.pi * W / 2).real
    pswf /= numpy.prod(
        numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
    )  # double factorial
    return pswf


def get_actual_work_terms(
    pswf, xM, xMxN_yP_size, yB_size, yN_size, xM_size, N, yP_size
):
    """
    Get gridding-related functions.
    Calculate actual work terms to use. We need both $n$ and $b$ in image space
    In case of gridding: "n": gridding function (except that we have it in image space here)
                         "b": grid correction function.

    :param pswf: prolate-spheroidal wave function
    :param xM: TODO ???
    :param xMxN_yP_size: length of the region to be cut out of the prepared facet data.
                         i.e. len(facet_m0_trunc)
    :param yB_size: effective facet size
    :param yN_size: needed padding
    :param xM_size: padded (rough) subgrid size
    :param N: total image size in one direction
    :param yP_size: padded (rough) image size (facet)

    :return:
        Fb: Fourier transform of grid correction function
        Fn: Fourier transform of gridding function
        facet_m0_trunc: mask truncated to a facet (image space)

    Note (Peter W): The reason they're single functions (i.e. we only compute one Fn, Fb and m0
    instead of one per facet/subgrid) is that we assume that they are all the same function,
    just shifted in grid and image space respectively (to the positions of the subgrids and facets)
    """
    Fb = 1 / extract_mid(pswf, yB_size)
    Fn = pswf[(yN_size // 2) % int(1 / 2 / xM) :: int(1 / 2 / xM)]
    facet_m0_trunc = pswf * numpy.sinc(coordinates(yN_size) * xM_size / N * yN_size)
    facet_m0_trunc = (
        xM_size
        * yP_size
        / N
        * extract_mid(ifft(pad_mid(facet_m0_trunc, yP_size)), xMxN_yP_size).real
    )
    return Fb, Fn, facet_m0_trunc


def generate_mask(n_image, ndata_point, true_usable_size, offset):
    """
    Determine the appropriate A/B masks for cutting the subgrid/facet out.
    We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

    :param n_image: total image size in one side (N)
    :param ndata_point: number of data points (nsubgrid or nfacet)
    :param true_usable_size: true usable size (xA_size or yB_size)
    :param offset: subgrid or facet offset (subgrid_off or facet_off)

    :return: mask: subgrid_A or facet_B
    """
    mask = numpy.zeros((ndata_point, true_usable_size), dtype=int)
    subgrid_border = (offset + numpy.hstack([offset[1:], [n_image + offset[0]]])) // 2
    for i in range(ndata_point):
        left = (subgrid_border[i - 1] - offset[i] + true_usable_size // 2) % n_image
        right = subgrid_border[i] - offset[i] + true_usable_size // 2
        assert (
            left >= 0 and right <= true_usable_size
        ), "xA / yB not large enough to cover subgrids / facets!"
        mask[i, left:right] = 1

    return mask
