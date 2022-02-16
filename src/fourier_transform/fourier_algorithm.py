"""Distributed Fourier Transform Module."""
import itertools

import scipy.special
import scipy.signal
import numpy
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
    result = mask_element * extract_mid(
        numpy.roll(true_image, offset_i, axis), true_usable_size
    )
    return result


def make_subgrid_and_facet(
    G,
    FG,
    sizes_class,
    subgrid_A,
    subgrid_off,
    facet_B,
    facet_off,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param subgrid_A: subgrid mask
    :param subgrid_off: subgrid offset
    :param facet_B: facet mask
    :param facet_off: facet offset
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty((sizes_class.nsubgrid, sizes_class.xA_size), dtype=complex)
        facet = numpy.empty((sizes_class.nfacet, sizes_class.yB_size), dtype=complex)

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(sizes_class.nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G,
                -subgrid_off[i],
                sizes_class.xA_size,
                subgrid_A[i],
                axis=None,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(sizes_class.nfacet):
            facet[j] = _ith_subgrid_facet_element(
                FG,
                -facet_off[j],
                sizes_class.yB_size,
                facet_B[j],
                axis=None,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty(
            (sizes_class.nsubgrid, sizes_class.nsubgrid, sizes_class.xA_size, sizes_class.xA_size),
            dtype=complex,
        )
        facet = numpy.empty(
            (sizes_class.nfacet, sizes_class.nfacet, sizes_class.yB_size, sizes_class.yB_size), dtype=complex
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(range(sizes_class.nsubgrid), range(sizes_class.nsubgrid)):
            subgrid[i0][i1] = _ith_subgrid_facet_element(
                G,
                (-subgrid_off[i0], -subgrid_off[i1]),
                sizes_class.xA_size,
                numpy.outer(subgrid_A[i0], subgrid_A[i1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(range(sizes_class.nfacet), range(sizes_class.nfacet)):
            facet[j0][j1] = _ith_subgrid_facet_element(
                FG,
                (-facet_off[j0], -facet_off[j1]),
                sizes_class.yB_size,
                numpy.outer(facet_B[j0], facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


def make_subgrid_and_facet_dask_array(
    G,
    FG,
    sizes_class,
    subgrid_A,
    subgrid_off,
    facet_B,
    facet_off,
):
    """
    Calculate the actual subgrids and facets. Same as make_subgrid_and_facet
    but implemented using dask.array. Consult that function for a full docstring.
    Only for 1D (2D not yet implemented).

    Returns a dask.array.
    """
    subgrid = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                G, -subgrid_off[i], sizes_class.xA_size, subgrid_A[i], axis=None
            )
            for i in range(sizes_class.nsubgrid)
        ],
        chunks=(1, sizes_class.xA_size),
    ).astype(complex)

    facet = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                FG, -facet_off[j], sizes_class.yB_size, facet_B[j], axis=None
            )
            for j in range(sizes_class.nfacet)
        ],
        chunks=(1, sizes_class.yB_size),
    ).astype(complex)

    return subgrid, facet


@dask_wrapper
def facet_contribution_to_subgrid_1d(
    BjFj,
    facet_m0_trunc,
    offset_i,
    Fn,
    sizes_class,
    **kwargs,
):
    """
    Extract the facet contribution to a subgrid for 1D version.

    :param BjFj: Prepared facet data (i.e. multiplied by b, padded to yP_size, Fourier transformed)
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param offset_i: ith offset value (subgrid)
    :param Fn: Fourier transform of gridding function
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

    :return: facet_in_a_subgrid: facet contribution to a subgrid
    """
    MiBjFj = facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * sizes_class.yP_size // sizes_class.N),
        sizes_class.xMxN_yP_size,
    )
    MiBjFj_sum = extract_mid(MiBjFj, sizes_class.xM_yP_size)
    MiBjFj_sum[: sizes_class.xN_yP_size // 2] = (
        MiBjFj_sum[: sizes_class.xN_yP_size // 2]
        + MiBjFj[-sizes_class.xN_yP_size // 2 :]
    )
    MiBjFj_sum[-sizes_class.xN_yP_size // 2 :] = (
        MiBjFj_sum[-sizes_class.xN_yP_size // 2 :]
        + MiBjFj[: sizes_class.xN_yP_size // 2 :]
    )

    facet_in_a_subgrid = Fn * extract_mid(fft(MiBjFj_sum), sizes_class.xM_yN_size)

    return facet_in_a_subgrid


def facet_contribution_to_subgrid_1d_dask_array(
    BjFj,
    facet_m0_trunc,
    offset_i,
    sizes_class,
    Fn,
):
    """
    Extract the facet contribution of a subgrid for 1D version.
    Same as facet_contribution_to_subgrid_1d but implemented using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    MiBjFj = facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * sizes_class.yP_size // sizes_class.N),
        sizes_class.xMxN_yP_size,
    ).rechunk(sizes_class.xMxN_yP_size)
    MiBjFj_sum = extract_mid(MiBjFj, sizes_class.xM_yP_size).rechunk(
        sizes_class.xM_yP_size
    )
    MiBjFj_sum[: sizes_class.xN_yP_size // 2] += MiBjFj[-sizes_class.xN_yP_size // 2 :]
    MiBjFj_sum[-sizes_class.xN_yP_size // 2 :] += MiBjFj[
        : sizes_class.xN_yP_size // 2 :
    ]

    facet_in_a_subgrid = Fn * extract_mid(
        fft(MiBjFj_sum), sizes_class.xM_yN_size
    ).rechunk(sizes_class.xM_yN_size)

    return facet_in_a_subgrid


@dask_wrapper
def prepare_facet_1d(facet_j, Fb, yP_size, **kwargs):
    """
    :param facet_j: jth facet element
    :param Fb: Fourier transform of grid correction function
    :param yP_size: padded (rough) facet size
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return ifft(pad_mid(facet_j * Fb, yP_size))  # prepare facet


def facets_to_subgrid_1d(
    facet,
    Fb,
    Fn,
    facet_m0_trunc,
    subgrid_off,
    sizes_class,
    dtype,
    use_dask,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.

    :param facet: numpy array of facets
    :param Fb: Fourier transform of grid correction function
    :param Fn: Fourier transform of gridding function
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param subgrid_off: subgrid offset
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param dtype: data type
    :param use_dask: use Dask?

    :return: RNjMiBjFj: array of contributions of this facet to different subgrids
    """
    RNjMiBjFj = numpy.empty((sizes_class.nsubgrid, sizes_class.nfacet, sizes_class.xM_yN_size), dtype=dtype)

    if use_dask:
        RNjMiBjFj = RNjMiBjFj.tolist()

    for j in range(sizes_class.nfacet):
        BjFj = prepare_facet_1d(
            facet[j], Fb, sizes_class.yP_size, use_dask=use_dask, nout=1
        )
        for i in range(sizes_class.nsubgrid):
            RNjMiBjFj[i][j] = facet_contribution_to_subgrid_1d(  # extract subgrid
                BjFj,
                facet_m0_trunc,
                subgrid_off[i],
                Fn,
                sizes_class,
                use_dask=use_dask,
                nout=1,
            )

    return RNjMiBjFj


def facets_to_subgrid_1d_dask_array(
    facet,
    Fb,
    Fn,
    facet_m0_trunc,
    subgrid_off,
    sizes_class,
    dtype,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.
    Same as facets_to_subgrid_1d but implemented using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    RNjMiBjFj = dask.array.from_array(
        [
            [
                facet_contribution_to_subgrid_1d_dask_array(
                    ifft(
                        pad_mid(facet[i] * Fb, sizes_class.yP_size).rechunk(
                            sizes_class.yP_size
                        )
                    ),  # prepare facet
                    facet_m0_trunc,
                    subgrid_off[j],
                    sizes_class,
                    Fn,
                )
                for i in range(sizes_class.nfacet)
            ]
            for j in range(sizes_class.nsubgrid)
        ],
        chunks=(1, 1, sizes_class.xM_yN_size),
    ).astype(dtype)
    return RNjMiBjFj


@dask_wrapper
def _add_padded_value(nmbf, facet_off_j, xM_size, N, **kwargs):
    """
    :param nmbf: a single element of the subgrid array calculated from
                 facets by facets_to_subgrid_1d
    :param facet_off_j: jth facet offset
    :param xM_size: padded (rough) subgrid size
    :param N: total image size
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return numpy.roll(pad_mid(nmbf, xM_size), facet_off_j * xM_size // N)


def reconstruct_subgrid_1d(
    nmbfs, facet_off, subgrid_A, sizes_class, use_dask
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
    :param facet_off: facet offset
    :param subgrid_A: subgrid mask
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param use_dask: use Dask?

    :return: approximate subgrid
    """
    approx_subgrid = numpy.ndarray((sizes_class.nsubgrid, sizes_class.xA_size), dtype=complex)
    if use_dask:
        approx_subgrid = approx_subgrid.tolist()
        approx = numpy.zeros((sizes_class.nsubgrid, sizes_class.xM_size), dtype=complex)
        approx = approx.tolist()
        for i in range(sizes_class.nsubgrid):
            for j in range(sizes_class.nfacet):
                approx[i] = approx[i] + _add_padded_value(
                    nmbfs[i][j],
                    facet_off[j],
                    sizes_class.xM_size,
                    sizes_class.N,
                    use_dask=use_dask,
                    nout=1,
                )
            # TODO: Here we used dask array in order to avoid complications of ifft, but this is not optimal.
            approx_array = dask.array.from_delayed(
                approx[i], (sizes_class.xM_size,), dtype=complex
            )
            approx_subgrid[i] = subgrid_A[i] * extract_mid(
                ifft(approx_array), sizes_class.xA_size
            )
    else:
        for i in range(sizes_class.nsubgrid):
            approx = numpy.zeros(sizes_class.xM_size, dtype=complex)
            for j in range(sizes_class.nfacet):
                approx = approx + _add_padded_value(
                    nmbfs[i, j], facet_off[j], sizes_class.xM_size, sizes_class.N
                )
            approx_subgrid[i, :] = subgrid_A[i] * extract_mid(
                ifft(approx), sizes_class.xA_size
            )

    return approx_subgrid


def reconstruct_subgrid_1d_dask_array(
    nmbfs, facet_off, subgrid_A, sizes_class
):
    """
    Reconstruct the subgrid array calculated by facets_to_subgrid_1d.
    Same as reconstruct_subgrid_1d but using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    approx = dask.array.from_array(
        numpy.zeros((sizes_class.nsubgrid, sizes_class.xM_size), dtype=complex),
        chunks=(1, sizes_class.xM_size),
    )
    approx_subgrid = dask.array.from_array(
        numpy.ndarray((sizes_class.nsubgrid, sizes_class.xA_size), dtype=complex),
        chunks=(1, sizes_class.xA_size),
    )
    for i in range(sizes_class.nsubgrid):
        for j in range(sizes_class.nfacet):
            padded = pad_mid(nmbfs[i, j, :], sizes_class.xM_size).rechunk(
                sizes_class.xM_size
            )
            approx[i, :] += numpy.roll(
                padded, facet_off[j] * sizes_class.xM_size // sizes_class.N
            )

        approx_subgrid[i, :] = subgrid_A[i] * extract_mid(
            ifft(approx[i]), sizes_class.xA_size
        ).rechunk(sizes_class.xA_size)

    return approx_subgrid


@dask_wrapper
def _calculate_fns_term(subgrid_ith, facet_off_jth, Fn, sizes_class, **kwargs):
    """
    :param subgrid_ith: ith subgrid element
    :param facet_off_jth: jth facet offset
    :param Fn: Fourier transform of gridding function
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return Fn * extract_mid(
        numpy.roll(
            fft(pad_mid(subgrid_ith, sizes_class.xM_size)),
            -facet_off_jth * sizes_class.xM_size // sizes_class.N,
        ),
        sizes_class.xM_yN_size,
    )


def subgrid_to_facet_1d(
    subgrid, facet_off, Fn, sizes_class, use_dask
):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.

    :param subgrid: numpy array of subgrids
    :param facet_off: facet offset
    :param Fn: Fourier transform of gridding function
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param use_dask: use Dask?

    :return: distributed facet array determined from input subgrid array
    """
    FNjSi = numpy.empty((sizes_class.nsubgrid, sizes_class.nfacet, sizes_class.xM_yN_size), dtype=complex)

    if use_dask:
        FNjSi = FNjSi.tolist()

    for i in range(sizes_class.nsubgrid):
        for j in range(sizes_class.nfacet):
            FNjSi[i][j] = _calculate_fns_term(
                subgrid[i],
                facet_off[j],
                Fn,
                sizes_class,
                use_dask=use_dask,
                nout=1,
            )

    return FNjSi


def subgrid_to_facet_1d_dask_array(
    subgrid, facet_off, Fn, sizes_class
):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.
    Same as subgrid_to_facet_1d but implemented with dask.array
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    FNjSi = dask.array.from_array(
        [
            [
                extract_mid(
                    numpy.roll(
                        fft(
                            pad_mid(subgrid[j], sizes_class.xM_size).rechunk(
                                sizes_class.xM_size
                            )
                        ),
                        -facet_off[i] * sizes_class.xM_size // sizes_class.N,
                    ),
                    sizes_class.xM_yN_size,
                )
                for i in range(sizes_class.nfacet)
            ]
            for j in range(sizes_class.nsubgrid)
        ]
    )
    distributed_facet = Fn * FNjSi

    return distributed_facet


@dask_wrapper
def add_subgrid_contribution_1d(
    nafs_ij,
    facet_m0_trunc,
    subgrid_off_i,
    sizes_class,
    **kwargs,
):
    """
    Add subgrid contribution to a single facet.

    :param nafs_ij: redistributed facet array [i:j] element
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param subgrid_off_i: subgrid offset [i]th element
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: subgrid contribution
    """
    NjSi = numpy.zeros(sizes_class.xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, sizes_class.xM_yP_size)
    NjSi_mid[:] = ifft(
        pad_mid(nafs_ij, sizes_class.xM_yP_size)
    )  # updates NjSi via reference!
    NjSi[-sizes_class.xN_yP_size // 2 :] = NjSi_mid[: sizes_class.xN_yP_size // 2]
    NjSi[: sizes_class.xN_yP_size // 2 :] = NjSi_mid[-sizes_class.xN_yP_size // 2 :]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(facet_m0_trunc * NjSi, sizes_class.yP_size),
            subgrid_off_i * sizes_class.yP_size // sizes_class.N,
        )
    )
    subgrid_contrib = extract_mid(FMiNjSi, sizes_class.yB_size)
    return subgrid_contrib


def add_subgrid_contribution_1d_dask_array(
    nafs_ij,
    facet_m0_trunc,
    subgrid_off_i,
    sizes_class,
):
    """
    Add subgrid contribution to a single facet.
    Same as add_subgrid_contribution_1d but implemented with dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    NjSi = numpy.zeros(sizes_class.xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, sizes_class.xM_yP_size)
    NjSi_mid[:] = ifft(
        pad_mid(nafs_ij, sizes_class.xM_yP_size).rechunk(sizes_class.xM_yP_size)
    )  # updates NjSi via reference!
    NjSi[-sizes_class.xN_yP_size // 2 :] = NjSi_mid[: sizes_class.xN_yP_size // 2]
    NjSi[: sizes_class.xN_yP_size // 2 :] = NjSi_mid[-sizes_class.xN_yP_size // 2 :]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(facet_m0_trunc * NjSi, sizes_class.yP_size),
            subgrid_off_i * sizes_class.yP_size // sizes_class.N,
        )
    )
    subgrid_contrib = extract_mid(FMiNjSi, sizes_class.yB_size)
    return subgrid_contrib


def reconstruct_facet_1d(
    nafs,
    facet_m0_trunc,
    subgrid_off,
    Fb,
    facet_B,
    sizes_class,
    use_dask=False,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d

    :param nafs: redistributed facet array calculated from facets by subgrid_to_facet_1d
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param subgrid_off: subgrid offset
    :param Fb: Fourier transform of grid correction function
    :param facet_B: facet mask
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param use_dask: whether to run with dask.delayed or not

    :return: approximate facet
    """
    # Note: compared to reconstruct_subgrid_1d, which does the same but for subgrids,
    # the order in which we perform calculations is different:
    # here, we apply fft then extract_mid to cut down from size yP_size to yB_size,
    # finally we do the sum (which results in approx);
    # In reconstruct_subgrid_1d we first do the sum, and then cut from xM_size to xA_size,
    # hence approx starts as a larger sized array in that case.

    approx_facet = numpy.ndarray((sizes_class.nfacet, sizes_class.yB_size), dtype=complex)
    if use_dask:
        approx_facet = approx_facet.tolist()
        approx = numpy.zeros((sizes_class.nfacet, sizes_class.yB_size), dtype=complex)
        approx = approx.tolist()
        for j in range(sizes_class.nfacet):
            for i in range(sizes_class.nsubgrid):
                approx[j] = approx[j] + add_subgrid_contribution_1d(
                    nafs[i][j],
                    facet_m0_trunc,
                    subgrid_off[i],
                    sizes_class,
                    use_dask=use_dask,
                    nout=1,
                )
            approx_facet[j] = approx[j] * Fb * facet_B[j]
    else:
        for j in range(sizes_class.nfacet):
            approx = numpy.zeros(sizes_class.yB_size, dtype=complex)
            for i in range(sizes_class.nsubgrid):
                approx = approx + add_subgrid_contribution_1d(
                    nafs[i, j],
                    facet_m0_trunc,
                    subgrid_off[i],
                    sizes_class,
                )
            approx_facet[j, :] = approx * Fb * facet_B[j]

    return approx_facet


def reconstruct_facet_1d_dask_array(
    nafs,
    facet_m0_trunc,
    subgrid_off,
    Fb,
    facet_B,
    sizes_class,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d.
    Same as reconstruct_facet_1d but implemented with dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    approx = dask.array.from_array(
        numpy.zeros((sizes_class.nfacet, sizes_class.yB_size), dtype=complex),
        chunks=(1, sizes_class.yB_size),
    )  # why is this not yP_size? for the subgrid version it uses xM_size
    approx_facet = dask.array.from_array(
        numpy.ndarray((sizes_class.nfacet, sizes_class.yB_size), dtype=complex),
        chunks=(1, sizes_class.yB_size),
    )
    for j in range(sizes_class.nfacet):
        for i in range(sizes_class.nsubgrid):
            approx[j, :] += add_subgrid_contribution_1d_dask_array(
                nafs[i, j],
                facet_m0_trunc,
                subgrid_off[i],
                sizes_class,
            )
        approx_facet[j, :] = approx[j] * Fb * facet_B[j]

    return approx_facet


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
    BF = pad_mid_a(facet * broadcast_a(Fb, len(facet.shape), axis), yP_size, axis)
    BF = ifft_a(BF, axis)
    return BF


@dask_wrapper
def extract_subgrid(
    BF,
    axis,
    subgrid_off_i,
    facet_m0_trunc,
    Fn,
    sizes_class,
    **kwargs,
):
    """
    Extract the facet contribution of a subgrid.
    TODO: maybe rename function to reflect this definition
        See discussion at https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/merge_requests/4#note_825003275

    :param BF:
    :param axis: Axis
    :param subgrid_off_i:
    :param facet_m0_trunc:
    :param Fn:
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return:
    """
    dims = len(BF.shape)
    BF_mid = extract_mid_a(
        numpy.roll(BF, -subgrid_off_i * sizes_class.yP_size // sizes_class.N, axis),
        sizes_class.xMxN_yP_size,
        axis,
    )
    MBF = broadcast_a(facet_m0_trunc, dims, axis) * BF_mid
    MBF_sum = numpy.array(extract_mid_a(MBF, sizes_class.xM_yP_size, axis))
    xN_yP_size = sizes_class.xMxN_yP_size - sizes_class.xM_yP_size
    # [:xN_yP_size//2] / [-xN_yP_size//2:] for axis, [:] otherwise
    slc1 = slice_a(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = slice_a(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    MBF_sum[slc1] += MBF[slc2]
    MBF_sum[slc2] += MBF[slc1]

    return broadcast_a(Fn, len(BF.shape), axis) * extract_mid_a(
        fft_a(MBF_sum, axis), sizes_class.xM_yN_size, axis
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
    return fft(pad_mid(subgrid, xM_size))


@dask_wrapper
def extract_facet_contribution(FSi, Fn, facet_off_j, sizes_class, axis, **kwargs):
    """
    Extract contribution of subgrid to a facet

    :param Fsi:
    :param Fn:
    :param facet_off_j:
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param axis: Axis
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: Contribution of facet on the subgrid

    """
    return broadcast_a(Fn, len(FSi.shape), axis) * extract_mid_a(
        numpy.roll(FSi, -facet_off_j * sizes_class.xM_size // sizes_class.N, axis),
        sizes_class.xM_yN_size,
        axis,
    )


@dask_wrapper
def add_subgrid_contribution(
    dims,
    NjSi,
    facet_m0_trunc,
    subgrid_off_i,
    sizes_class,
    axis,
    **kwargs,
):
    """
    Add subgrid contribution to a facet

    :param MiNjSi_sum:
    :param NjSi:
    :param facet_m0_trunc:
    :param subgrid_off:
    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param axis:
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return MiNjSi_sum:

    """
    xN_yP_size = sizes_class.xMxN_yP_size - sizes_class.xM_yP_size
    NjSi_mid = ifft_a(pad_mid_a(NjSi, sizes_class.xM_yP_size, axis), axis)
    NjSi_temp = pad_mid_a(NjSi_mid, sizes_class.xMxN_yP_size, axis)
    slc1 = slice_a(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = slice_a(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    NjSi_temp[slc1] = NjSi_mid[slc2]
    NjSi_temp[slc2] = NjSi_mid[slc1]
    NjSi_temp = NjSi_temp * broadcast_a(facet_m0_trunc, len(NjSi.shape), axis)

    return numpy.roll(
        pad_mid_a(NjSi_temp, sizes_class.yP_size, axis),
        subgrid_off_i * sizes_class.yP_size // sizes_class.N,
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
    return extract_mid_a(fft_a(MiNjSi_sum, axis), yB_size, axis) * broadcast_a(
        Fb * facet_B_j, len(MiNjSi_sum.shape), axis
    )


# COMMON 1D and 2D FUNCTIONS -- SETTING UP FOR ALGORITHM TO RUN
def calculate_pswf(sizes_class, alpha):
    """
    Calculate PSWF (prolate-spheroidal wave function) at the
    full required resolution (facet size)

    :param sizes_class: Sizes class object containing fundamental and derived parameters
    :param alpha: TODO: ???, int
    """
    pswf = anti_aliasing_function(
        sizes_class.yN_size, alpha, numpy.pi * sizes_class.W / 2
    ).real
    pswf /= numpy.prod(
        numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
    )  # double factorial
    return pswf


def get_actual_work_terms(pswf, sizes_class):
    """
    Get gridding-related functions.
    Calculate actual work terms to use. We need both $n$ and $b$ in image space
    In case of gridding: "n": gridding function (except that we have it in image space here)
                         "b": grid correction function.

    :param pswf: prolate-spheroidal wave function
    :param sizes_class: Sizes class object containing fundamental and derived parameters

    :return:
        Fb: Fourier transform of grid correction function
        Fn: Fourier transform of gridding function
        facet_m0_trunc: mask truncated to a facet (image space)

    Note (Peter W): The reason they're single functions (i.e. we only compute one Fn, Fb and m0
    instead of one per facet/subgrid) is that we assume that they are all the same function,
    just shifted in grid and image space respectively (to the positions of the subgrids and facets)
    """
    Fb = 1 / extract_mid(pswf, sizes_class.yB_size)
    Fn = pswf[
        (sizes_class.yN_size // 2)
        % int(1 / 2 / sizes_class.xM) :: int(1 / 2 / sizes_class.xM)
    ]
    facet_m0_trunc = pswf * numpy.sinc(
        coordinates(sizes_class.yN_size)
        * sizes_class.xM_size
        / sizes_class.N
        * sizes_class.yN_size
    )
    facet_m0_trunc = (
        sizes_class.xM_size
        * sizes_class.yP_size
        / sizes_class.N
        * extract_mid(
            ifft(pad_mid(facet_m0_trunc, sizes_class.yP_size)), sizes_class.xMxN_yP_size
        ).real
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
