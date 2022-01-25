"""Distributed Fourier Transform Module."""
import os

import scipy.special
import scipy.signal
import numpy

import dask.array


USE_DASK = os.getenv("USE_DASK", False) == "True"


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


def fft(a, axis=None):
    """Fourier transformation from image to grid space

    :param a: image in `lm` coordinate space
    :param axis: int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
        (doc from numpy.fft.ifftshift docstring)
    :returns: `uv` grid
    """
    if len(a.shape) == 1:
        return numpy.fft.fftshift(numpy.fft.fft(numpy.fft.ifftshift(a)))
    elif len(a.shape) == 2:
        return numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(a)))
    assert False, "Unsupported image shape for FFT!"


def ifft(a):
    """Fourier transformation from grid to image space

    :param a: `uv` grid to transform
    :returns: an image in `lm` coordinate space
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

    # 2D Prolate spheroidal angular function is seperable
    return numpy.outer(
        anti_aliasing_function(shape[0], m, c), anti_aliasing_function(shape[1], m, c)
    )


# 1D FOURIER ALGORITHM FUNCTIONS
def make_subgrid_and_facet(
    G,
    nsubgrid,
    xA_size,
    subgrid_A,
    subgrid_off,
    nfacet,
    yB_size,
    facet_B,
    facet_off,
):
    """

    Calculate the actual subgrids and facets

    :param G: x
    :param nsubgrid: Number of subgrid
    :param xA_size: x
    :param subgrid_A: Subgrid A
    :param subgrid_off: Subgrid off
    :param nfacet: Number of facet
    :param yB_size: Effective facet size
    :param facet_B: Facet B
    :param facet_off: Facet off

    :return: subgrid and facet

    """
    FG = fft(G)

    if USE_DASK:
        subgrid = dask.array.from_array(
            [
                _ith_subgrid_facet_element(G, subgrid_A[i], subgrid_off[i], xA_size)
                for i in range(nsubgrid)
            ],
            chunks=(1, xA_size),
        ).astype(complex)

        facet = dask.array.from_array(
            [
                _ith_subgrid_facet_element(FG, facet_B[j], facet_off[j], yB_size)
                for j in range(nfacet)
            ],
            chunks=(1, yB_size),
        ).astype(complex)

    else:
        subgrid = numpy.empty((nsubgrid, xA_size), dtype=complex)
        for i in range(nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G, subgrid_A[i], subgrid_off[i], xA_size
            )

        facet = numpy.empty((nfacet, yB_size), dtype=complex)
        for j in range(nfacet):
            facet[j] = _ith_subgrid_facet_element(FG, facet_B[j], facet_off[j], yB_size)

    return subgrid, facet


def _ith_subgrid_facet_element(true_image, mask_i, offset_i, true_usable_size):
    result = mask_i * extract_mid(numpy.roll(true_image, -offset_i), true_usable_size)
    return result


def subgrid_range_1(
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
    Subgrid range 1

    param BjFj:
    param i:
    param facet_m0_trunc:
    param subgrid_off:
    param yP_size: Facet size, padded for m convolution (internal)
    param N: Total image size on a side
    param xMxN_yP_size:
    param xN_yP_size:
    param xM_yP_size:
    param xM_yN_size:
    param Fn:

    return:

    """
    if USE_DASK:
        MiBjFj = facet_m0_trunc * extract_mid(
            numpy.roll(BjFj, -offset_i * yP_size // N), xMxN_yP_size
        ).rechunk(xMxN_yP_size)
        MiBjFj_sum = extract_mid(MiBjFj, xM_yP_size).rechunk(xM_yP_size)
        MiBjFj_sum[: xN_yP_size // 2] += MiBjFj[-xN_yP_size // 2 :]
        MiBjFj_sum[-xN_yP_size // 2 :] += MiBjFj[: xN_yP_size // 2 :]

        result = Fn * extract_mid(fft(MiBjFj_sum), xM_yN_size).rechunk(xM_yN_size)

    else:
        MiBjFj = facet_m0_trunc * extract_mid(
            numpy.roll(BjFj, -offset_i * yP_size // N), xMxN_yP_size
        )
        MiBjFj_sum = extract_mid(MiBjFj, xM_yP_size)
        MiBjFj_sum[: xN_yP_size // 2] += MiBjFj[-xN_yP_size // 2 :]
        MiBjFj_sum[-xN_yP_size // 2 :] += MiBjFj[: xN_yP_size // 2 :]

        result = Fn * extract_mid(fft(MiBjFj_sum), xM_yN_size)

    return result


def facets_to_subgrid_1(
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

    Facet to subgrid 1.

    param facet: Facet
    param nsubgrid: Number of subgrid
    param nfacet: Number of facet
    param xM_yN_size:
    param Fb:
    param Fn:
    param yP_size:
    param facet_m0_trunc:
    param subgrid_off: Subgrid off
    param N: Total image size on a side
    param xMxN_yP_size:
    param xN_yP_size:
    param xM_yP_size:
    param dtype: D type

    return: RNjMiBjFj

    """
    if USE_DASK:
        RNjMiBjFj = dask.array.from_array(
            [
                [
                    subgrid_range_1(
                        ifft(pad_mid(facet[i] * Fb, yP_size).rechunk(yP_size)),
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
        # TODO: compute should not happen here; for now dask is implemented until this point
        RNjMiBjFj = RNjMiBjFj.compute()

    else:
        RNjMiBjFj = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=dtype)
        for j in range(nfacet):
            BjFj = ifft(pad_mid(facet[j] * Fb, yP_size))
            for i in range(nsubgrid):
                RNjMiBjFj[i, j] = subgrid_range_1(
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
                )

    return RNjMiBjFj


def facets_to_subgrid_2(
    nmbfs, xM_size, nfacet, facet_off, N, subgrid_A, xA_size, nsubgrid
):
    """
    Facet to subgrid 2

    param nmbfs:
    param i:
    param xM_size:
    param nfacet: Number of facet
    param facet_off: Facet off
    param N: total image size on a side
    param subgrid_A: Subgrid A
    param xA_size:

    return:
    """
    result = numpy.ndarray((nsubgrid, xA_size), dtype=complex)
    for i in range(nsubgrid):
        approx = numpy.zeros(xM_size, dtype=complex)
        for j in range(nfacet):
            approx += numpy.roll(
                pad_mid(nmbfs[i, j], xM_size), facet_off[j] * xM_size // N
            )
        result[i, :] = subgrid_A[i] * extract_mid(ifft(approx), xA_size)

    return result


def subgrid_to_facet_1(
    subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
):
    """
    Subgrid to facet 1

    param subgrid: Subgtid
    param nsubgrid: Number of subgrid
    param nfacet: Number of facet
    param xM_yN_size:
    param xM_size:
    param facet_off: Facet off
    param N: Total image size on a side
    param Fn:

    return:
    """
    if USE_DASK:
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
        result = Fn * FNjSi
        # TODO: compute should not happen here; for now dask is implemented until this point
        result = result.compute()

    else:
        FNjSi = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=complex)
        for i in range(nsubgrid):
            FSi = fft(pad_mid(subgrid[i], xM_size))
            for j in range(nfacet):
                FNjSi[i, j] = extract_mid(
                    numpy.roll(FSi, -facet_off[j] * xM_size // N), xM_yN_size
                )
        result = Fn * FNjSi

    return result


def subgrid_to_facet_2(
    nafs,
    j,
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

    Subgrid to facet 2

    param nafs:
    param j:
    param yB_size: Effective facet size
    param nsubgrid:
    param xMxN_yP_size:
    param xM_yP_size:
    param xN_yP_size:
    param facet_m0_trunc:
    param yP_size: Facet size, padded for m convolution (internal)
    param subgrid_off: Subgrid off
    param N: Total image size on a side
    param Fb:
    param facet_B: Facet B

    return:
    """
    approx = numpy.zeros(yB_size, dtype=complex)
    for i in range(nsubgrid):
        approx += subgrid_range_2(
            xMxN_yP_size,
            xM_yP_size,
            i,
            j,
            nafs,
            xN_yP_size,
            facet_m0_trunc,
            yP_size,
            subgrid_off,
            yB_size,
            N,
        )

    return approx * Fb * facet_B[j]


def subgrid_range_2(
    xMxN_yP_size,
    xM_yP_size,
    i,
    j,
    nafs,
    xN_yP_size,
    facet_m0_trunc,
    yP_size,
    subgrid_off,
    yB_size,
    N,
):
    """
    Subgrid Range 2

    param xMxN_yP_size:
    param xM_yP_size:
    param i:
    param j:
    param nafs:
    param xN_yP_size:
    param facet_m0_trunc:
    param yP_size: Facet size, padded for m convolution (internal)
    param subgrid_off: Subgrid off
    param yB_size: Effective facet size
    param N: Total image size on a side

    return:
    """
    NjSi = numpy.zeros(xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, xM_yP_size)
    NjSi_mid[:] = ifft(pad_mid(nafs[i, j], xM_yP_size))  # updates NjSi via reference!
    NjSi[-xN_yP_size // 2 :] = NjSi_mid[: xN_yP_size // 2]
    NjSi[: xN_yP_size // 2 :] = NjSi_mid[-xN_yP_size // 2 :]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(facet_m0_trunc * NjSi, yP_size), subgrid_off[i] * yP_size // N
        )
    )
    return extract_mid(FMiNjSi, yB_size)


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

    param BF:
    param i:
    param axis: Axis
    param subgrid_off:
    param yP_size: Facet size, padded for m convolution (internal)
    param xMxN_yP_size:
    param facet_m0_trunc:
    param xM_yP_size:
    param Fn:
    param xM_yN_size:
    param N: Total image size on a side

    return:
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
    param subgrid: Subgrid
    param xM_size: Subgrid size, padded for transfer (internal)

    return: the FS term
    """
    return fft(pad_mid(subgrid, xM_size))


def extract_facet_contribution(FSi, Fn, facet_off, j, xM_size, N, xM_yN_size, axis):
    """
    Extract contribution of subgrid to a facet

    param Fsi:
    param Fn:
    param facet_off:
    param j: Index on the facet
    param xM_size: Subgrid size, padded for transfer (internal)
    param N: Total image size on a side
    param xM_yN_size:
    param axis: Axis

    return: Contribution of facet on the subgrid

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

    param MiNjSi_sum:
    param NjSi:
    param i:
    param facet_m0_trunc:
    param subgrid_off:
    param xMxN_yP_size:
    param xM_yP_size:
    param yP_size:
    param N:
    param axis:

    return MiNjSi_sum:

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

    param MiNjSi_sum:
    param Fb:
    param facet_B:
    param yB_size:
    param j:
    param axis:

    return: The finished facet (in BMNAF term)
    """
    return extract_mid_a(fft_a(MiNjSi_sum, axis), yB_size, axis) * broadcast_a(
        Fb * facet_B[j], len(MiNjSi_sum.shape), axis
    )


# COMMON 1D and 2D FUNCTIONS -- SETTING UP FOR ALGORITHM TO RUN

# TODO: what exactly are the things we return here? + needs docstring
def get_actual_work_terms(
    pswf, xM, xMxN_yP_size, yB_size, yN_size, xM_size, N, yP_size
):
    # Calculate actual work terms to use. We need both $n$ and $b$ in image space
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


def calculate_pswf(yN_size, alpha, W):
    # Calculate PSWF at the full required resolution (facet size)
    pswf = anti_aliasing_function(yN_size, alpha, numpy.pi * W / 2).real
    pswf /= numpy.prod(
        numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
    )  # double factorial
    return pswf


def generate_mask(n_image, ndata_point, true_usable_size, offset):
    """
    :param n_image: N
    :param ndata_point: nsubgrid, nfacet
    :param true_usable_size: xA_size, yB_size
    :param offset: subgrid_off, facet_off
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

    return mask  # subgrid_A, facet_B
