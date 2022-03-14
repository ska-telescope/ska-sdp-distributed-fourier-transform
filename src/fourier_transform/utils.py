import itertools
import logging
import numpy
from matplotlib import pylab, patches as patches

from src.fourier_transform.fourier_algorithm import (
    coordinates,
    extract_mid_along_axis,
    ifft_along_axis,
    pad_mid_along_axis,
    prepare_facet,
    extract_subgrid,
    fft_along_axis,
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
    """
    Helper for marking ranges in a graph.

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
        lbl_y = (y0**7 * y1) ** (1 / 8)
    ax.annotate(lbl, (x1 + x_offset * wdt, lbl_y))


def display_plots(x, legend=None, grid=False, xlim=None, fig_name=None):
    """
    Display plots using pylab

    :param x: X values
    :param legend: Legend
    :param grid: Grid
    :param xlim: X axis limitation
    :param fig_name: partial name or prefix (can include path) if figure is saved
                     if None, pylab.show() is called instead
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
def plot_1(pswf, constants_class, fig_name=None):
    """
    :param pswf: prolate-spheroidal wave function
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure is saved
                     if None, pylab.show() is called instead
    """
    xN = constants_class.W / constants_class.yN_size / 2
    yN = constants_class.yN_size / 2
    yB = constants_class.yB_size / 2
    xN_size = constants_class.N * constants_class.W / constants_class.yN_size

    pylab.clf()
    pylab.semilogy(
        coordinates(4 * int(xN_size)) * 4 * xN_size / constants_class.N,
        extract_mid_along_axis(
            numpy.abs(ifft_along_axis(pad_mid_along_axis(pswf, constants_class.N, axis=0), axis=0)),
            4 * int(xN_size),
            axis=0,
        ),
    )
    pylab.legend(["n"])
    mark_range("$x_n$", -xN, xN)
    pylab.xlim(
        -2 * int(xN_size) / constants_class.N,
        (2 * int(xN_size) - 1) / constants_class.N,
    )
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_n.png")

    pylab.clf()
    pylab.semilogy(coordinates(constants_class.yN_size) * constants_class.yN_size, pswf)
    pylab.legend(["$\\mathcal{F}[n]$"])
    mark_range("$y_B$", -yB, yB)
    pylab.xlim(-constants_class.N // 2, constants_class.N // 2 - 1)
    mark_range("$y_n$", -yN, yN)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_fn.png")


# TODO: needs better name; used both for 1D and 2D
def plot_2(constants_class, fig_name=None):
    """
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure is saved
                     if None, pylab.show() is called instead
    """
    pylab.clf()
    pylab.semilogy(
        coordinates(constants_class.xMxN_yP_size)
        / constants_class.yP_size
        * constants_class.xMxN_yP_size,
        constants_class.facet_m0_trunc,
    )
    xM = constants_class.xM_size / 2 / constants_class.N
    mark_range("xM", -xM, xM)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_xm.png")


def calculate_and_plot_errors_subgrid_1d(
    approx_subgrid, nsubgrid, subgrid, xA_size, N, to_plot=True, fig_name=None
):
    """
    Facet to subgrid error terms. Log and plot them.

    :param approx_subgrid:
    :param nsubgrid:
    :param subgrid:
    :param xA_size:
    :param N:
    :param to_plot: plot results?
    :param fig_name: partial name or prefix (can include path) if figure is saved
                     if None, pylab.show() is called instead
    """
    # Let us look at the error terms:
    if to_plot:
        xA = xA_size / 2 / N

        pylab.clf()
        fig = pylab.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)

    err_sum = 0
    err_sum_img = 0
    for i in range(nsubgrid):
        error = approx_subgrid[i] - subgrid[i]
        if to_plot:
            ax1.semilogy(xA * 2 * coordinates(xA_size), numpy.abs(error))
            ax2.semilogy(N * coordinates(xA_size), numpy.abs(fft_along_axis(error, axis=0)))
        err_sum += numpy.abs(error) ** 2 / nsubgrid
        err_sum_img += numpy.abs(fft_along_axis(error, axis=0)) ** 2 / nsubgrid

    if to_plot:
        #  By feeding the implementation single-pixel inputs we can create a full error map.
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
    approx_facet, facet, constants_class, to_plot=True, fig_name=None
):
    if to_plot:
        pylab.clf()
        fig = pylab.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)

    err_sum = 0
    err_sum_img = 0
    for j in range(constants_class.nfacet):
        error = approx_facet[j] - facet[j]
        err_sum += numpy.abs(ifft_along_axis(error, axis=0)) ** 2
        err_sum_img += numpy.abs(error) ** 2
        if to_plot:
            ax1.semilogy(coordinates(constants_class.yB_size), numpy.abs(ifft_along_axis(error, axis=0)))
            ax2.semilogy(
                constants_class.yB_size * coordinates(constants_class.yB_size),
                numpy.abs(error),
            )

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_sum)),
        numpy.sqrt(numpy.mean(err_sum_img)),
    )

    if to_plot:
        xA = constants_class.xA_size / 2 / constants_class.N
        yB = constants_class.yB_size / 2
        xM = constants_class.xM_size / 2 / constants_class.N

        mark_range("$x_A$", -xA, xA, ax=ax1)
        mark_range("$x_M$", -xM, xM, ax=ax1)
        mark_range("$y_B$", -yB, yB, ax=ax2)
        mark_range("$0.5$", -0.5, 0.5, ax=ax1)
        if fig_name is None:
            pylab.show()
        else:
            pylab.savefig(f"{fig_name}_error_subgrid_to_facet_1d.png")


def _plot_error(err_mean, err_mean_img, fig_name, direction):
    """
    :param direction: either "facet_to_subgrid" or "subgrid_to_facet"; used for plot naming
    """
    pylab.clf()
    pylab.figure(figsize=(16, 8))
    pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10))
    pylab.colorbar()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_error_mean_{direction}_2d.png")
    pylab.clf()
    pylab.figure(figsize=(16, 8))
    pylab.imshow(numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10))
    pylab.colorbar()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_error_mean_image_{direction}_2d.png")


def errors_facet_to_subgrid_2d(
    NMBF_NMBF,
    constants_class,
    subgrid_2,
    to_plot=True,
    fig_name=None,
):
    err_mean = 0
    err_mean_img = 0
    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        approx = numpy.zeros(
            (constants_class.xM_size, constants_class.xM_size), dtype=complex
        )
        for j0, j1 in itertools.product(
            range(constants_class.nfacet), range(constants_class.nfacet)
        ):
            padded_nmbf = pad_mid_along_axis(
                pad_mid_along_axis(
                    NMBF_NMBF[i0, i1, j0, j1], constants_class.xM_size, axis=0
                ),
                constants_class.xM_size,
                axis=1,
            )
            approx += numpy.roll(
                padded_nmbf,
                (
                    constants_class.facet_off[j0]
                    * constants_class.xM_size
                    // constants_class.N,
                    constants_class.facet_off[j1]
                    * constants_class.xM_size
                    // constants_class.N,
                ),
                (0, 1),
            )
        approx = extract_mid_along_axis(
            extract_mid_along_axis(ifft_along_axis(ifft_along_axis(approx, axis=0), axis=1), constants_class.xA_size, axis=0),
            constants_class.xA_size,
            axis=1,
        )
        approx *= numpy.outer(
            constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
        )
        err_mean += (
            numpy.abs(approx - subgrid_2[i0, i1]) ** 2 / constants_class.nsubgrid**2
        )
        err_mean_img += (
            numpy.abs(fft_along_axis(fft_along_axis(approx - subgrid_2[i0, i1], axis=0), axis=1) ** 2
                      / constants_class.nsubgrid ** 2
                      ))

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "facet_to_subgrid")


def errors_subgrid_to_facet_2d(
    BMNAF_BMNAF, facet_2, nfacet, yB_size, to_plot=True, fig_name=None
):
    err_mean = 0
    err_mean_img = 0
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        approx = numpy.zeros((yB_size, yB_size), dtype=complex)
        approx += BMNAF_BMNAF[j0, j1]

        err_mean += numpy.abs(ifft_along_axis(ifft_along_axis(approx - facet_2[j0, j1], axis=0), axis=1)) ** 2 / nfacet ** 2
        err_mean_img += numpy.abs(approx - facet_2[j0, j1]) ** 2 / nfacet**2

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "subgrid_to_facet")


# TODO: refactor this; it repeats a lot of code from the 2D case - what's the difference?
def test_accuracy_facet_to_subgrid(
    constants_class,
    xs=252,
    ys=252,
    to_plot=True,
    fig_name=None,
):
    """
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param xs:
    :param ys:
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix into PNG files.
                     If to_plot is set to False, fig_name doesn't have an effect.
    """
    subgrid_2 = numpy.empty(
        (
            constants_class.nsubgrid,
            constants_class.nsubgrid,
            constants_class.xA_size,
            constants_class.xA_size,
        ),
        dtype=complex,
    )
    facet_2 = numpy.empty(
        (
            constants_class.nfacet,
            constants_class.nfacet,
            constants_class.yB_size,
            constants_class.yB_size,
        ),
        dtype=complex,
    )

    FG_2 = numpy.zeros((constants_class.N, constants_class.N))
    FG_2[ys, xs] = 1
    G_2 = ifft_along_axis(ifft_along_axis(FG_2, axis=0), axis=1)

    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        subgrid_2[i0, i1] = extract_mid_along_axis(
            extract_mid_along_axis(
                numpy.roll(
                    G_2,
                    (
                        -constants_class.subgrid_off[i0],
                        -constants_class.subgrid_off[i1],
                    ),
                    (0, 1),
                ),
                constants_class.xA_size,
                axis=0,
            ),
            constants_class.xA_size,
            axis=1,
        )
        subgrid_2[i0, i1] *= numpy.outer(
            constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
        )
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        facet_2[j0, j1] = extract_mid_along_axis(
            extract_mid_along_axis(
                numpy.roll(
                    FG_2,
                    (-constants_class.facet_off[j0], -constants_class.facet_off[j1]),
                    (0, 1),
                ),
                constants_class.yB_size,
                axis=0,
            ),
            constants_class.yB_size,
            axis=1,
        )
        facet_2[j0, j1] *= numpy.outer(
            constants_class.facet_B[j0], constants_class.facet_B[j1]
        )

    NMBF_NMBF = numpy.empty(
        (
            constants_class.nsubgrid,
            constants_class.nsubgrid,
            constants_class.nfacet,
            constants_class.nfacet,
            constants_class.xM_yN_size,
            constants_class.xM_yN_size,
        ),
        dtype=complex,
    )
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        BF_F = prepare_facet(
            facet_2[j0, j1], 0, constants_class.Fb, constants_class.yP_size
        )
        BF_BF = prepare_facet(BF_F, 1, constants_class.Fb, constants_class.yP_size)
        for i0 in range(constants_class.nsubgrid):
            NMBF_BF = extract_subgrid(
                BF_BF,
                0,
                constants_class.subgrid_off[i0],
                constants_class,
            )
            for i1 in range(constants_class.nsubgrid):
                NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                    NMBF_BF,
                    1,
                    constants_class.subgrid_off[i1],
                    constants_class,
                )

    err_mean = err_mean_img = 0
    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        approx = numpy.zeros(
            (constants_class.xM_size, constants_class.xM_size), dtype=complex
        )
        for j0, j1 in itertools.product(
            range(constants_class.nfacet), range(constants_class.nfacet)
        ):
            approx += numpy.roll(
                pad_mid_along_axis(
                    pad_mid_along_axis(
                        NMBF_NMBF[i0, i1, j0, j1], constants_class.xM_size, axis=0
                    ),
                    constants_class.xM_size,
                    axis=1,
                ),
                (
                    constants_class.facet_off[j0]
                    * constants_class.xM_size
                    // constants_class.N,
                    constants_class.facet_off[j1]
                    * constants_class.xM_size
                    // constants_class.N,
                ),
                (0, 1),
            )
        approx = extract_mid_along_axis(
            extract_mid_along_axis(ifft_along_axis(ifft_along_axis(approx, axis=0), axis=1), constants_class.xA_size, axis=0),
            constants_class.xA_size,
            axis=1,
        )
        approx *= numpy.outer(
            constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
        )
        err_mean += (
            numpy.abs(approx - subgrid_2[i0, i1]) ** 2 / constants_class.nsubgrid**2
        )
        err_mean_img += (
                numpy.abs(fft_along_axis(fft_along_axis(approx - subgrid_2[i0, i1], axis=0), axis=1)) ** 2
                / constants_class.nsubgrid ** 2
        )
    x = numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10)

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        if fig_name:
            full_name = f"{fig_name}_test_accuracy_facet_to_subgrid_2d"
        else:
            full_name = None
        display_plots(x, fig_name=full_name)


def test_accuracy_subgrid_to_facet(
    constants_class,
    xs=252,
    ys=252,
    to_plot=True,
    fig_name=None,
):
    """
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param xs:
    :param ys:
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix into PNG files.
                     If to_plot is set to False, fig_name doesn't have an effect.
    """
    subgrid_2 = numpy.empty(
        (
            constants_class.nsubgrid,
            constants_class.nsubgrid,
            constants_class.xA_size,
            constants_class.xA_size,
        ),
        dtype=complex,
    )
    facet_2 = numpy.empty(
        (
            constants_class.nfacet,
            constants_class.nfacet,
            constants_class.yB_size,
            constants_class.yB_size,
        ),
        dtype=complex,
    )

    FG_2 = numpy.zeros((constants_class.N, constants_class.N))
    FG_2[ys, xs] = 1
    G_2 = ifft_along_axis(ifft_along_axis(FG_2, axis=0), axis=1)

    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        subgrid_2[i0, i1] = extract_mid_along_axis(
            extract_mid_along_axis(
                numpy.roll(
                    G_2,
                    (
                        -constants_class.subgrid_off[i0],
                        -constants_class.subgrid_off[i1],
                    ),
                    (0, 1),
                ),
                constants_class.xA_size,
                axis=0,
            ),
            constants_class.xA_size,
            axis=1,
        )
        subgrid_2[i0, i1] *= numpy.outer(
            constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
        )
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        facet_2[j0, j1] = extract_mid_along_axis(
            extract_mid_along_axis(
                numpy.roll(
                    FG_2,
                    (-constants_class.facet_off[j0], -constants_class.facet_off[j1]),
                    (0, 1),
                ),
                constants_class.yB_size,
                axis=0,
            ),
            constants_class.yB_size,
            axis=1,
        )
        facet_2[j0, j1] *= numpy.outer(
            constants_class.facet_B[j0], constants_class.facet_B[j1]
        )

    NAF_NAF = numpy.empty(
        (
            constants_class.nsubgrid,
            constants_class.nsubgrid,
            constants_class.nfacet,
            constants_class.nfacet,
            constants_class.xM_yN_size,
            constants_class.xM_yN_size,
        ),
        dtype=complex,
    )
    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        AF_AF = prepare_subgrid(subgrid_2[i0, i1], constants_class.xM_size)
        for j0 in range(constants_class.nfacet):
            NAF_AF = extract_facet_contribution(
                AF_AF, constants_class.facet_off[j0], constants_class, 0
            )
            for j1 in range(constants_class.nfacet):
                NAF_NAF[i0, i1, j0, j1] = extract_facet_contribution(
                    NAF_AF, constants_class.facet_off[j1], constants_class, 1
                )

    BMNAF_BMNAF = numpy.empty(
        (
            constants_class.nfacet,
            constants_class.nfacet,
            constants_class.yB_size,
            constants_class.yB_size,
        ),
        dtype=complex,
    )
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        MNAF_BMNAF = numpy.zeros(
            (constants_class.yP_size, constants_class.yB_size), dtype=complex
        )
        for i0 in range(constants_class.nsubgrid):
            NAF_MNAF = numpy.zeros(
                (constants_class.xM_yN_size, constants_class.yP_size), dtype=complex
            )
            for i1 in range(constants_class.nsubgrid):
                NAF_MNAF = NAF_MNAF + add_subgrid_contribution(
                    len(NAF_MNAF.shape),
                    NAF_NAF[i0, i1, j0, j1],
                    constants_class.subgrid_off[i1],
                    constants_class,
                    1,
                )
            NAF_BMNAF = finish_facet(
                NAF_MNAF,
                constants_class.Fb,
                constants_class.facet_B[j1],
                constants_class.yB_size,
                1,
            )
            MNAF_BMNAF = MNAF_BMNAF + add_subgrid_contribution(
                len(MNAF_BMNAF.shape),
                NAF_BMNAF,
                constants_class.subgrid_off[i0],
                constants_class,
                0,
            )
        BMNAF_BMNAF[j0, j1] = finish_facet(
            MNAF_BMNAF,
            constants_class.Fb,
            constants_class.facet_B[j0],
            constants_class.yB_size,
            0,
        )

    pylab.rcParams["figure.figsize"] = 16, 8
    err_mean = err_mean_img = 0

    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        approx = numpy.zeros(
            (constants_class.yB_size, constants_class.yB_size), dtype=complex
        )
        approx += BMNAF_BMNAF[j0, j1]
        err_mean += (
                numpy.abs(ifft_along_axis(ifft_along_axis(approx - facet_2[j0, j1], axis=0), axis=1)) ** 2 / constants_class.nfacet ** 2
        )
        err_mean_img += (
            numpy.abs(approx - facet_2[j0, j1]) ** 2 / constants_class.nfacet**2
        )

    x = numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10)
    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        if fig_name:
            full_name = f"{fig_name}_test_accuracy_subgrid_to_facet_2d"
        else:
            full_name = None
        display_plots(x, fig_name=full_name)
