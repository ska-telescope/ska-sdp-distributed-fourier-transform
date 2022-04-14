# pylint: disable=too-many-locals, too-many-arguments, unused-argument
"""
Utility Functions

We provide functions that help plotting and
basic validation of the algorithm.
"""

import itertools
import logging

import numpy
from matplotlib import patches, pylab

from src.fourier_transform.dask_wrapper import dask_wrapper
from src.fourier_transform.fourier_algorithm import (
    coordinates,
    extract_mid,
    fft,
    ifft,
    pad_mid,
)

log = logging.getLogger("fourier-logger")


# PLOTTING UTILS
def mark_range(
    lbl,
    x0,
    x1=None,
    y0=None,
    y1=None,
    ax=None,
    x_offset=1 / 200,
    linestyle="--",
):
    """
    Helper for marking ranges in a graph.

    :param lbl: graph label
    :param x0: X axis lower limit
    :param x1: X axis upper limit
    :param y0: Y axis lower limit
    :param y1: Y axis upper limit
    :param ax: Axes
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
        patches.PathPatch(
            patches.Path([(x0, y0), (x0, y1)]), linestyle=linestyle
        )
    )
    if x1 is not None:
        ax.add_patch(
            patches.PathPatch(
                patches.Path([(x1, y0), (x1, y1)]), linestyle=linestyle
            )
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
    :param grid: If true, construct Grid
    :param xlim: X axis limit (a tuple that contains lower and upper limits)
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
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


def plot_pswf(constants_class, fig_name=None):
    """
    Plot to check that PSWF indeed satisfies intended bounds.

    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
    """
    xN = constants_class.W / constants_class.yN_size / 2
    yN = constants_class.yN_size / 2
    yB = constants_class.yB_size / 2
    xN_size = constants_class.N * constants_class.W / constants_class.yN_size

    pylab.clf()
    pylab.semilogy(
        coordinates(4 * int(xN_size)) * 4 * xN_size / constants_class.N,
        extract_mid(
            numpy.abs(
                ifft(
                    pad_mid(constants_class.pswf, constants_class.N, axis=0),
                    axis=0,
                )
            ),
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
    pylab.semilogy(
        coordinates(constants_class.yN_size) * constants_class.yN_size,
        constants_class.pswf,
    )
    pylab.legend(["$\\mathcal{F}[n]$"])
    mark_range("$y_B$", -yB, yB)
    pylab.xlim(-constants_class.N // 2, constants_class.N // 2 - 1)
    mark_range("$y_n$", -yN, yN)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_fn.png")


def plot_work_terms(constants_class, fig_name=None):
    """
    Calculate actual work terms to use and plot to check them.
    We need both n and b in image space.

    TODO: What exactly are these work terms??
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
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


def _plot_error(err_mean, err_mean_img, fig_name, direction):
    """
    Plot the error terms.

    :param err_mean: Mean of the errors, real part
    :param err_mean_img: Mean of the errors, imaginary part
    :param fig_name: Name of the figure
    :param direction: either "facet_to_subgrid" or "subgrid_to_facet"
                      used for plot naming
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
    """
    Calculate the error terms for the 2D facet to subgrid algorithm.

    :param NMBF_NMBF: array of individual facet contributions
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param subgrid_2: 2D numpy array of subgrids
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.

    """
    err_mean = 0
    err_mean_img = 0
    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        approx = numpy.zeros(
            (constants_class.xA_size, constants_class.xA_size), dtype=complex
        )
        approx += NMBF_NMBF[i0, i1]

        err_mean += (
            numpy.abs(approx - subgrid_2[i0, i1]) ** 2
            / constants_class.nsubgrid**2
        )
        err_mean_img += numpy.abs(
            fft(fft(approx - subgrid_2[i0, i1], axis=0), axis=1) ** 2
            / constants_class.nsubgrid**2
        )

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "facet_to_subgrid")


def errors_subgrid_to_facet_2d(
    BMNAF_BMNAF, facet_2, constants_class, to_plot=True, fig_name=None
):
    """
    Calculate the error terms for the 2D subgrid to facet algorithm.

    :param BMNAF_BMNAF: array of individual subgrid contributions
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param facet_2: 2D numpy array of facets
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    """
    err_mean = 0
    err_mean_img = 0
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        approx = numpy.zeros(
            (constants_class.yB_size, constants_class.yB_size), dtype=complex
        )
        approx += BMNAF_BMNAF[j0, j1]

        err_mean += (
            numpy.abs(ifft(ifft(approx - facet_2[j0, j1], axis=0), axis=1))
            ** 2
            / constants_class.nfacet**2
        )
        err_mean_img += (
            numpy.abs(approx - facet_2[j0, j1]) ** 2
            / constants_class.nfacet**2
        )

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "subgrid_to_facet")


@dask_wrapper
def add_two(one, two, **kwargs):
    """
    Functions for iterative operations to
    accelerate the construction of graphs

    :param one: array or graph
    :param two: array or graph

    :returns: added object
    """
    if one is None:
        return two
    return one + two
