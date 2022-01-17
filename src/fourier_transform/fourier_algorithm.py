"""Distributed Fourier Transform Module."""

import matplotlib.patches as patches
from matplotlib import pylab
import scipy.special
import scipy.signal
import numpy
import itertools

# Fixing seed of numpy random
numpy.random.seed(123456789)


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


def fmt(x):
    """

    :param x: X

    :return: x
    """
    if x >= 1024 * 1024 and (x % (1024 * 1024)) == 0:
        return "%dM" % (x // 1024 // 1024)
    if x >= 1024 and (x % 1024) == 0:
        return "%dk" % (x // 1024)
    return "%d" % x


def mark_range(
    lbl, x0, x1=None, y0=None, y1=None, ax=None, x_offset=1 / 200, linestyle="--"
):
    """Helper for marking ranges in a graph.

    :param lbl: label
    :param x0: X0
    :param x1: X1
    :param y1: Y1
    :param ax: Ax
    :param x_offset: X offset
    :param linestyle: Linestyle

    """
    if ax is None:
        ax = pylab.gca()
    if y0 is None:
        y0 = ax.get_ylim()[1]
    if y1 is None:
        y1 = ax.get_ylim()[0]
    wdt = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.add_patch(
        patches.PathPatch(patches.Path([(x0, y0), (x0, y1)]), linestyle=linestyle)
    )
    if x1 is not None:
        ax.add_patch(
            patches.PathPatch(patches.Path([(x1, y0), (x1, y1)]), linestyle=linestyle)
        )
    else:
        x1 = x0
    if pylab.gca().get_yscale() == "linear":
        lbl_y = (y0 * 7 + y1) / 8
    else:
        # Some type of log scale
        lbl_y = (y0 ** 7 * y1) ** (1 / 8)
    ax.annotate(lbl, (x1 + x_offset * wdt, lbl_y))


def find_x_sorted_smooth(xs, ys, y):
    """Find sorted smooth.

    :param xs: Xs
    :param ys: Ys
    :param y: Y

    :return: xs

    """

    assert len(xs) == len(ys)
    pos = numpy.searchsorted(ys, y)
    if pos <= 0:
        return xs[0]
    if pos >= len(ys) or ys[pos] == ys[pos - 1]:
        return xs[len(ys) - 1]
    w = (y - ys[pos - 1]) / (ys[pos] - ys[pos - 1])
    return xs[pos - 1] * (1 - w) + xs[pos] * w


def find_x_sorted_logsmooth(xs, ys, y):
    """
    Find sorted log smooth.

    :param xs: Xs
    :param ys: Ys
    :param y: Y

    :return: log xs

    """
    return find_x_sorted_smooth(xs, numpy.log(numpy.maximum(1e-100, ys)), numpy.log(y))


def whole(xs):
    """."""
    return numpy.all(numpy.abs(xs - numpy.around(xs)) < 1e-13)


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
    subgrid = numpy.empty((nsubgrid, xA_size), dtype=complex)
    for i in range(nsubgrid):
        subgrid[i] = subgrid_A[i] * extract_mid(numpy.roll(G, -subgrid_off[i]), xA_size)
    facet = numpy.empty((nfacet, yB_size), dtype=complex)
    for j in range(nfacet):
        facet[j] = facet_B[j] * extract_mid(numpy.roll(FG, -facet_off[j]), yB_size)
    return subgrid, facet


def subgrid_range_1(
    BjFj,
    i,
    facet_m0_trunc,
    subgrid_off,
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
    MiBjFj = facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -subgrid_off[i] * yP_size // N), xMxN_yP_size
    )
    MiBjFj_sum = numpy.array(extract_mid(MiBjFj, xM_yP_size))
    MiBjFj_sum[: xN_yP_size // 2] += MiBjFj[-xN_yP_size // 2 :]
    MiBjFj_sum[-xN_yP_size // 2 :] += MiBjFj[: xN_yP_size // 2 :]

    return Fn * extract_mid(fft(MiBjFj_sum), xM_yN_size)


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
    RNjMiBjFj = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=dtype)
    for j in range(nfacet):
        BjFj = ifft(pad_mid(facet[j] * Fb, yP_size))
        for i in range(nsubgrid):
            RNjMiBjFj[i, j] = subgrid_range_1(
                BjFj,
                i,
                facet_m0_trunc,
                subgrid_off,
                yP_size,
                N,
                xMxN_yP_size,
                xN_yP_size,
                xM_yP_size,
                xM_yN_size,
                Fn,
            )
    return RNjMiBjFj


def facets_to_subgrid_2(nmbfs, i, xM_size, nfacet, facet_off, N, subgrid_A, xA_size):
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
    approx = numpy.zeros(xM_size, dtype=complex)
    for j in range(nfacet):
        approx += numpy.roll(pad_mid(nmbfs[i, j], xM_size), facet_off[j] * xM_size // N)
    return subgrid_A[i] * extract_mid(ifft(approx), xA_size)


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
    FNjSi = numpy.empty((nsubgrid, nfacet, xM_yN_size), dtype=complex)
    for i in range(nsubgrid):
        FSi = fft(pad_mid(subgrid[i], xM_size))
        for j in range(nfacet):
            FNjSi[i, j] = extract_mid(
                numpy.roll(FSi, -facet_off[j] * xM_size // N), xM_yN_size
            )
    return Fn * FNjSi


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
    slc = [numpy.newaxis] * dims
    slc[axis] = slice(None)
    return a[slc]


# TODO Two functions with same name?
def broadcast_a(a, dims, axis):
    """
    Broadcast A

    param a: A
    param dims: Dimensions
    param axis: Axis


    return:
    """
    return a[slice_a(numpy.newaxis, slice(None), dims, axis)]


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


def test_accuracy(
    nsubgrid,
    xA_size,
    nfacet,
    yB_size,
    N,
    subgrid_off,
    subgrid_A,
    facet_off,
    facet_B,
    xM_yN_size,
    xM_size,
    Fb,
    yP_size,
    xMxN_yP_size,
    facet_m0_trunc,
    xM_yP_size,
    Fn,
    xs=252,
    ys=252,
):
    """

    param nsubgrid: Number of subgrids
    param xA_size: Effective subgrid size
    param nfacet: Number of facets
    param yB_size: Effective facet size
    param N: Total image size on a side
    param subgrid_off: Subgrid off
    param subgrid_A: Subgrid A
    param facet_off: Facet off
    param facet_B: Facet B
    param xM_yN_size:
    param xM_size: Subgrid size, padded for transfer (internal)
    param Fb:
    param yP_size: Facet size, padded for m convolution (internal)
    param xMxN_yP_size:
    param facet_m0_trunc:
    param xM_yP_size:
    param Fn:
    param xs:
    param ys:
    """
    subgrid_2 = numpy.empty((nsubgrid, nsubgrid, xA_size, xA_size), dtype=complex)
    facet_2 = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)

    # G_2 = numpy.exp(2j*numpy.pi*numpy.random.rand(N,N))*numpy.random.rand(N,N)/2
    # FG_2 = fft(G_2)

    FG_2 = numpy.zeros((N, N))
    FG_2[ys, xs] = 1
    G_2 = ifft(FG_2)

    for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
        subgrid_2[i0, i1] = extract_mid(
            numpy.roll(G_2, (-subgrid_off[i0], -subgrid_off[i1]), (0, 1)), xA_size
        )
        subgrid_2[i0, i1] *= numpy.outer(subgrid_A[i0], subgrid_A[i1])
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        facet_2[j0, j1] = extract_mid(
            numpy.roll(FG_2, (-facet_off[j0], -facet_off[j1]), (0, 1)), yB_size
        )
        facet_2[j0, j1] *= numpy.outer(facet_B[j0], facet_B[j1])

    NMBF_NMBF = numpy.empty(
        (nsubgrid, nsubgrid, nfacet, nfacet, xM_yN_size, xM_yN_size), dtype=complex
    )
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        BF_F = prepare_facet(facet_2[j0, j1], 0, Fb, yP_size)
        BF_BF = prepare_facet(BF_F, 1, Fb, yP_size)
        for i0 in range(nsubgrid):
            NMBF_BF = extract_subgrid(
                BF_BF,
                i0,
                0,
                subgrid_off,
                yP_size,
                xMxN_yP_size,
                facet_m0_trunc,
                xM_yP_size,
                Fn,
                xM_yN_size,
                N,
            )
            for i1 in range(nsubgrid):
                NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                    NMBF_BF,
                    i1,
                    1,
                    subgrid_off,
                    yP_size,
                    xMxN_yP_size,
                    facet_m0_trunc,
                    xM_yP_size,
                    Fn,
                    xM_yN_size,
                    N,
                )

    err_mean = err_mean_img = 0
    for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
        approx = numpy.zeros((xM_size, xM_size), dtype=complex)
        for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
            approx += numpy.roll(
                pad_mid(NMBF_NMBF[i0, i1, j0, j1], xM_size),
                (facet_off[j0] * xM_size // N, facet_off[j1] * xM_size // N),
                (0, 1),
            )
        approx = extract_mid(ifft(approx), xA_size)
        approx *= numpy.outer(subgrid_A[i0], subgrid_A[i1])
        err_mean += numpy.abs(approx - subgrid_2[i0, i1]) ** 2 / nsubgrid ** 2
        err_mean_img += numpy.abs(fft(approx - subgrid_2[i0, i1])) ** 2 / nsubgrid ** 2
    # pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10)); pylab.colorbar(); pylab.show()
    x = numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10)
    display_plots(x)
    print(
        "RMSE:",
        numpy.sqrt(numpy.mean(err_mean)),
        "(image:",
        numpy.sqrt(numpy.mean(err_mean_img)),
        ")",
    )


def display_plots(x, legend=None, grid=False, xlim=None):
    """Display plots using pylab

    param x: X values
    param legend: Legend
    param grid: Grid
    param xlim: X axis limitation
    """
    pylab.rcParams["figure.figsize"] = 16, 8
    pylab.rcParams["image.cmap"] = "viridis"
    if grid:
        pylab.grid()
    if legend is not None:
        pylab.legend(legend)
    if xlim is not None:
        pylab.xlim(xlim)
    pylab.imshow(x)
    pylab.colorbar()
    pylab.show()
