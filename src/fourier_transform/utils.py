import itertools
import logging
import numpy
from matplotlib import pylab, patches as patches

from src.fourier_transform.fourier_algorithm import (
    coordinates,
    extract_mid,
    ifft,
    pad_mid,
    prepare_facet,
    extract_subgrid,
    fft,
    prepare_subgrid,
    extract_facet_contribution,
    add_subgrid_contribution,
    finish_facet,
)

log = logging.getLogger("fourier-logger")


# MANUAL TESTING UTILS
def whole(xs):
    """."""
    return numpy.all(numpy.abs(xs - numpy.around(xs)) < 1e-13)


# PLOTTING UTILS
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


def display_plots(x, legend=None, grid=False, xlim=None, fig_name=None):
    """Display plots using pylab

    param x: X values
    param legend: Legend
    param grid: Grid
    param xlim: X axis limitation
    """
    pylab.clf()
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
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}.png")


# TODO: needs better name; used both for 1D and 2D
def plot_1(pswf, xN, xN_size, yB, yN, N, yN_size, fig_name=None):
    pylab.clf()
    pylab.semilogy(
        coordinates(4 * int(xN_size)) * 4 * xN_size / N,
        extract_mid(numpy.abs(ifft(pad_mid(pswf, N))), 4 * int(xN_size)),
    )
    pylab.legend(["n"])
    mark_range("$x_n$", -xN, xN)
    pylab.xlim(-2 * int(xN_size) / N, (2 * int(xN_size) - 1) / N)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_n.png")

    pylab.clf()
    pylab.semilogy(coordinates(yN_size) * yN_size, pswf)
    pylab.legend(["$\\mathcal{F}[n]$"])
    mark_range("$y_B$", -yB, yB)
    pylab.xlim(-N // 2, N // 2 - 1)
    mark_range("$y_n$", -yN, yN)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_fn.png")


# TODO: needs better name; used both for 1D and 2D
def plot_2(facet_m0_trunc, xM, xMxN_yP_size, yP_size, fig_name=None):
    pylab.clf()
    pylab.semilogy(coordinates(xMxN_yP_size) / yP_size * xMxN_yP_size, facet_m0_trunc)
    mark_range("xM", -xM, xM)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_xm.png")


def calculate_and_plot_errors_subgrid_1d(
    approx_subgrid, nsubgrid, subgrid, xA, xA_size, N, to_plot=True, fig_name=None
):
    # Let us look at the error terms:
    if to_plot:
        pylab.clf()
        fig = pylab.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)

    err_sum = 0
    err_sum_img = 0
    for i in range(nsubgrid):
        error = approx_subgrid[i] - subgrid[i]
        if to_plot:
            ax1.semilogy(xA * 2 * coordinates(xA_size), numpy.abs(error))
            ax2.semilogy(N * coordinates(xA_size), numpy.abs(fft(error)))
        err_sum += numpy.abs(error) ** 2 / nsubgrid
        err_sum_img += numpy.abs(fft(error)) ** 2 / nsubgrid

    if to_plot:
        mark_range("$x_A$", -xA, xA, ax=ax1)
        pylab.grid()
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_error_facet_to_subgrid_1d.png")

        pylab.clf()
        mark_range("$N/2$", -N / 2, N / 2, ax=ax2)
        pylab.grid()
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_empty_n_per_2_1d.png")

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_sum)),
        numpy.sqrt(numpy.mean(err_sum_img)),
    )


def calculate_and_plot_errors_facet_1d(
    approx_facet, facet, nfacet, xA, xM, yB, yB_size, to_plot=True, fig_name=None
):
    if to_plot:
        pylab.clf()
        fig = pylab.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)

    err_sum = 0
    err_sum_img = 0
    for j in range(nfacet):
        error = approx_facet[j] - facet[j]
        err_sum += numpy.abs(ifft(error)) ** 2
        err_sum_img += numpy.abs(error) ** 2
        if to_plot:
            ax1.semilogy(coordinates(yB_size), numpy.abs(ifft(error)))
            ax2.semilogy(yB_size * coordinates(yB_size), numpy.abs(error))

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_sum)),
        numpy.sqrt(numpy.mean(err_sum_img)),
    )

    if to_plot:
        mark_range("$x_A$", -xA, xA, ax=ax1)
        mark_range("$x_M$", -xM, xM, ax=ax1)
        mark_range("$y_B$", -yB, yB, ax=ax2)
        mark_range("$0.5$", -0.5, 0.5, ax=ax1)
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_error_subgrid_to_facet_1d.png")


def calculate_and_plot_errors_2d(
    NMBF_NMBF,
    facet_off,
    nfacet,
    nsubgrid,
    subgrid_2,
    subgrid_A,
    xM_size,
    N,
    xA_size,
    to_plot=True,
    fig_name=None,
):
    err_mean = 0
    err_mean_img = 0
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

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        pylab.clf()
        pylab.figure(figsize=(16, 8))
        pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10))
        pylab.colorbar()
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_error_mean_facet_to_subgrid_2d.png")

        pylab.clf()
        pylab.figure(figsize=(16, 8))
        pylab.imshow(numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10))
        pylab.colorbar()
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_error_mean_image_facet_to_subgrid_2d.png")


# TODO: refactor this; it repeats a lot of code from the 2D case - what's the difference?
def test_accuracy_facet_to_subgrid(
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
    to_plot=True,
    fig_name=None,
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
    x = numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10)

    if to_plot:
        full_name = f"{fig_name}_test_accuracy_facet_to_subgrid_2d"
        display_plots(x, fig_name=full_name)

    print(
        "RMSE:",
        numpy.sqrt(numpy.mean(err_mean)),
        "(image:",
        numpy.sqrt(numpy.mean(err_mean_img)),
        ")",
    )


def test_accuracy_subgrid_to_facet(
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

    NAF_NAF = numpy.empty(
        (nsubgrid, nsubgrid, nfacet, nfacet, xM_yN_size, xM_yN_size), dtype=complex
    )
    for i0, i1 in itertools.product(range(nsubgrid), range(nsubgrid)):
        AF_AF = prepare_subgrid(subgrid_2[i0, i1], xM_size)
        for j0 in range(nfacet):
            NAF_AF = extract_facet_contribution(
                AF_AF, Fn, facet_off, j0, xM_size, N, xM_yN_size, 0
            )
            for j1 in range(nfacet):
                NAF_NAF[i0, i1, j0, j1] = extract_facet_contribution(
                    NAF_AF, Fn, facet_off, j1, xM_size, N, xM_yN_size, 1
                )

    BMNAF_BMNAF = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        MNAF_BMNAF = numpy.zeros((yP_size, yB_size), dtype=complex)
        for i0 in range(nsubgrid):
            NAF_MNAF = numpy.zeros((xM_yN_size, yP_size), dtype=complex)
            for i1 in range(nsubgrid):
                add_subgrid_contribution(
                    NAF_MNAF,
                    NAF_NAF[i0, i1, j0, j1],
                    i1,
                    facet_m0_trunc,
                    subgrid_off,
                    xMxN_yP_size,
                    xM_yP_size,
                    yP_size,
                    N,
                    1,
                )
            NAF_BMNAF = finish_facet(NAF_MNAF, Fb, facet_B, yB_size, j1, 1)
            add_subgrid_contribution(
                MNAF_BMNAF,
                NAF_BMNAF,
                i0,
                facet_m0_trunc,
                subgrid_off,
                xMxN_yP_size,
                xM_yP_size,
                yP_size,
                N,
                0,
            )
        BMNAF_BMNAF[j0, j1] = finish_facet(MNAF_BMNAF, Fb, facet_B, yB_size, j0, 0)

    pylab.rcParams["figure.figsize"] = 16, 8
    err_mean = err_mean_img = 0

    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        approx = numpy.zeros((yB_size, yB_size), dtype=complex)
        approx += BMNAF_BMNAF[j0, j1]
        err_mean += numpy.abs(ifft(approx - facet_2[j0, j1])) ** 2 / nfacet ** 2
        err_mean_img += numpy.abs(approx - facet_2[j0, j1]) ** 2 / nfacet ** 2

    x = numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10)
    display_plots(x)  # pylab.savefig("plot_test_accuracy_subgrid_to_facet_2d.png")
    print(
        "RMSE:",
        numpy.sqrt(numpy.mean(err_mean)),
        "(image:",
        numpy.sqrt(numpy.mean(err_mean_img)),
        ")",
    )
