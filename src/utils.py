# pylint: disable=too-many-locals, too-many-arguments, unused-argument
"""
Utility Functions

We provide functions that help plotting and
basic validation of the algorithm.
"""

import itertools
import logging
import os
from http import client

import h5py
import numpy
from distributed import Lock
from matplotlib import patches, pylab

from src.fourier_transform.dask_wrapper import dask_wrapper
from src.fourier_transform.fourier_algorithm import (
    coordinates,
    extract_mid,
    fft,
    ifft,
    pad_mid,
    roll_and_extract_mid,
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


def plot_pswf(pswf, constants_class, fig_name=None):
    """
    Plot to check that PSWF indeed satisfies intended bounds.

    :param pswf: prolate-spheroidal wave function
    :param constants_class: BaseArrays or DistributedFFT class object
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
            numpy.abs(ifft(pad_mid(pswf, constants_class.N, axis=0), axis=0)),
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
        coordinates(constants_class.yN_size) * constants_class.yN_size, pswf
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
    :param constants_class: BaseArrays or DistributedFFT class object
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
    :param constants_class: BaseArrays or SparseFourierTransform class object
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
    :param constants_class: BaseArrays or SparseFourierTransform class object
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


# TODO: do we need this test? If not, need to refactor
def test_accuracy_facet_to_subgrid(
    sparse_ft_class,
    xs=252,
    ys=252,
    to_plot=True,
    fig_name=None,
):
    """
    Test the accuracy of the 2D facet to subgrid algorithm.
    This can be a a stand-alone test routine.
    The test creates an arbitrary FG term and plot the errors.

    :param sparse_ft_class: SparseFourierTransform class object
    :param xs: size of test image in X
    :param ys: size of test image in Y
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    """
    subgrid_2 = numpy.empty(
        (
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.xA_size,
            sparse_ft_class.xA_size,
        ),
        dtype=complex,
    )
    facet_2 = numpy.empty(
        (
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.yB_size,
            sparse_ft_class.yB_size,
        ),
        dtype=complex,
    )

    FG_2 = numpy.zeros((sparse_ft_class.N, sparse_ft_class.N))
    FG_2[ys, xs] = 1
    G_2 = ifft(ifft(FG_2, axis=0), axis=1)

    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        subgrid_2[i0, i1] = extract_mid(
            extract_mid(
                numpy.roll(
                    G_2,
                    (
                        -sparse_ft_class.subgrid_off[i0],
                        -sparse_ft_class.subgrid_off[i1],
                    ),
                    (0, 1),
                ),
                sparse_ft_class.xA_size,
                axis=0,
            ),
            sparse_ft_class.xA_size,
            axis=1,
        )
        subgrid_2[i0, i1] *= numpy.outer(
            sparse_ft_class.subgrid_A[i0], sparse_ft_class.subgrid_A[i1]
        )
    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        facet_2[j0, j1] = extract_mid(
            extract_mid(
                numpy.roll(
                    FG_2,
                    (
                        -sparse_ft_class.facet_off[j0],
                        -sparse_ft_class.facet_off[j1],
                    ),
                    (0, 1),
                ),
                sparse_ft_class.yB_size,
                axis=0,
            ),
            sparse_ft_class.yB_size,
            axis=1,
        )
        facet_2[j0, j1] *= numpy.outer(
            sparse_ft_class.facet_B[j0], sparse_ft_class.facet_B[j1]
        )

    NMBF_NMBF = numpy.empty(
        (
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.xM_yN_size,
            sparse_ft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        BF_F = sparse_ft_class.prepare_facet(facet_2[j0, j1], 0)
        BF_BF = sparse_ft_class.prepare_facet(BF_F, 1)
        for i0 in range(sparse_ft_class.nsubgrid):
            NMBF_BF = sparse_ft_class.extract_facet_contrib_to_subgrid(
                BF_BF,
                0,
                sparse_ft_class.subgrid_off[i0],
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[
                    i0, i1, j0, j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    1,
                    sparse_ft_class.subgrid_off[i1],
                )

    err_mean = err_mean_img = 0
    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        approx = numpy.zeros(
            (sparse_ft_class.xM_size, sparse_ft_class.xM_size), dtype=complex
        )
        for j0, j1 in itertools.product(
            range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
        ):
            approx += numpy.roll(
                pad_mid(
                    pad_mid(
                        NMBF_NMBF[i0, i1, j0, j1],
                        sparse_ft_class.xM_size,
                        axis=0,
                    ),
                    sparse_ft_class.xM_size,
                    axis=1,
                ),
                (
                    sparse_ft_class.facet_off[j0]
                    * sparse_ft_class.xM_size
                    // sparse_ft_class.N,
                    sparse_ft_class.facet_off[j1]
                    * sparse_ft_class.xM_size
                    // sparse_ft_class.N,
                ),
                (0, 1),
            )
        approx = extract_mid(
            extract_mid(
                ifft(ifft(approx, axis=0), axis=1),
                sparse_ft_class.xA_size,
                axis=0,
            ),
            sparse_ft_class.xA_size,
            axis=1,
        )
        approx *= numpy.outer(
            sparse_ft_class.subgrid_A[i0], sparse_ft_class.subgrid_A[i1]
        )
        err_mean += (
            numpy.abs(approx - subgrid_2[i0, i1]) ** 2
            / sparse_ft_class.nsubgrid**2
        )
        err_mean_img += (
            numpy.abs(fft(fft(approx - subgrid_2[i0, i1], axis=0), axis=1))
            ** 2
            / sparse_ft_class.nsubgrid**2
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
    sparse_ft_class,
    xs=252,
    ys=252,
    to_plot=True,
    fig_name=None,
):
    """
    Test the accuracy of the 2D subgrid to facet algorithm.
    This can be a stand-alone test routine.
    The test creates an arbitrary FG term and plot the errors.

    :param sparse_ft_class: SparseFourierTransform class object
    :param xs: size of test image in X
    :param ys: size of test image in X
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this
                     prefix into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    """
    subgrid_2 = numpy.empty(
        (
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.xA_size,
            sparse_ft_class.xA_size,
        ),
        dtype=complex,
    )
    facet_2 = numpy.empty(
        (
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.yB_size,
            sparse_ft_class.yB_size,
        ),
        dtype=complex,
    )

    FG_2 = numpy.zeros((sparse_ft_class.N, sparse_ft_class.N))
    FG_2[ys, xs] = 1
    G_2 = ifft(ifft(FG_2, axis=0), axis=1)

    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        subgrid_2[i0, i1] = extract_mid(
            extract_mid(
                numpy.roll(
                    G_2,
                    (
                        -sparse_ft_class.subgrid_off[i0],
                        -sparse_ft_class.subgrid_off[i1],
                    ),
                    (0, 1),
                ),
                sparse_ft_class.xA_size,
                axis=0,
            ),
            sparse_ft_class.xA_size,
            axis=1,
        )
        subgrid_2[i0, i1] *= numpy.outer(
            sparse_ft_class.subgrid_A[i0], sparse_ft_class.subgrid_A[i1]
        )
    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        facet_2[j0, j1] = extract_mid(
            extract_mid(
                numpy.roll(
                    FG_2,
                    (
                        -sparse_ft_class.facet_off[j0],
                        -sparse_ft_class.facet_off[j1],
                    ),
                    (0, 1),
                ),
                sparse_ft_class.yB_size,
                axis=0,
            ),
            sparse_ft_class.yB_size,
            axis=1,
        )
        facet_2[j0, j1] *= numpy.outer(
            sparse_ft_class.facet_B[j0], sparse_ft_class.facet_B[j1]
        )

    NAF_NAF = numpy.empty(
        (
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.xM_yN_size,
            sparse_ft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        AF_AF = sparse_ft_class.prepare_subgrid(subgrid_2[i0, i1])
        for j0 in range(sparse_ft_class.nfacet):
            NAF_AF = sparse_ft_class.extract_subgrid_contrib_to_facet(
                AF_AF, sparse_ft_class.facet_off[j0], 0
            )
            for j1 in range(sparse_ft_class.nfacet):
                NAF_NAF[
                    i0, i1, j0, j1
                ] = sparse_ft_class.extract_subgrid_contrib_to_facet(
                    NAF_AF, sparse_ft_class.facet_off[j1], 1
                )

    BMNAF_BMNAF = numpy.empty(
        (
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.yB_size,
            sparse_ft_class.yB_size,
        ),
        dtype=complex,
    )
    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        MNAF_BMNAF = numpy.zeros(
            (sparse_ft_class.yP_size, sparse_ft_class.yB_size), dtype=complex
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NAF_MNAF = numpy.zeros(
                (sparse_ft_class.xM_yN_size, sparse_ft_class.yP_size),
                dtype=complex,
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                NAF_MNAF = NAF_MNAF + sparse_ft_class.add_subgrid_contribution(
                    len(NAF_MNAF.shape),
                    NAF_NAF[i0, i1, j0, j1],
                    sparse_ft_class.subgrid_off[i1],
                    1,
                )
            NAF_BMNAF = sparse_ft_class.finish_facet(
                NAF_MNAF,
                sparse_ft_class.facet_B[j1],
                1,
            )
            MNAF_BMNAF = MNAF_BMNAF + sparse_ft_class.add_subgrid_contribution(
                len(MNAF_BMNAF.shape),
                NAF_BMNAF,
                sparse_ft_class.subgrid_off[i0],
                0,
            )
        BMNAF_BMNAF[j0, j1] = sparse_ft_class.finish_facet(
            MNAF_BMNAF,
            sparse_ft_class.facet_B[j0],
            0,
        )

    pylab.rcParams["figure.figsize"] = 16, 8
    err_mean = err_mean_img = 0

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        approx = numpy.zeros(
            (sparse_ft_class.yB_size, sparse_ft_class.yB_size), dtype=complex
        )
        approx += BMNAF_BMNAF[j0, j1]
        err_mean += (
            numpy.abs(ifft(ifft(approx - facet_2[j0, j1], axis=0), axis=1))
            ** 2
            / sparse_ft_class.nfacet**2
        )
        err_mean_img += (
            numpy.abs(approx - facet_2[j0, j1]) ** 2
            / sparse_ft_class.nfacet**2
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


@dask_wrapper
def add_two(one, two, **kwargs):
    """Functions for iterative operations to
           accelerate the construction of graphs

    :param one: array or graph
    :param two: array or graph

    :returns: added object
    """
    if one is None:
        return two
    return one + two


def make_G_2_FG_2(sparse_ft_class):
    """Generate standard data G and FG
        Memory may not be enough at larger
        scales

    :param sparse_ft_class: sparse_ft_class

    :returns: G,FG
    """
    log.info("\n== Generate A/B masks and subgrid/facet offsets")
    # Determine subgrid/facet offsets and the appropriate
    # A/B masks for cutting them out.
    # We are aiming for full coverage here:
    #   Every pixel is part of exactly one subgrid / facet.

    # adding sources
    add_sources = True
    if add_sources:
        FG_2 = numpy.zeros((sparse_ft_class.N, sparse_ft_class.N))
        source_count = 1000
        sources = [
            (
                numpy.random.randint(
                    -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                ),
                numpy.random.randint(
                    -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                ),
                numpy.random.rand()
                * sparse_ft_class.N
                * sparse_ft_class.N
                / numpy.sqrt(source_count)
                / 2,
            )
            for _ in range(source_count)
        ]
        for x, y, i in sources:
            FG_2[y + sparse_ft_class.N // 2, x + sparse_ft_class.N // 2] += i
        G_2 = ifft(ifft(FG_2, axis=0), axis=1)
    else:
        # without sources
        G_2 = (
            numpy.exp(
                2j
                * numpy.pi
                * numpy.random.rand(sparse_ft_class.N, sparse_ft_class.N)
            )
            * numpy.random.rand(sparse_ft_class.N, sparse_ft_class.N)
            / 2
        )
        FG_2 = fft(fft(G_2, axis=0), axis=1)

    log.info("Mean grid absolute: %s", numpy.mean(numpy.abs(G_2)))
    return G_2, FG_2


@dask_wrapper
def DFT_chunk_work(G_2_path, chunk_slice, sources, chunksize, N, **kwargs):
    """
    Calculate the value of a chunk of the DFT and write to hdf5

    :param G_2_path: the hdf5 file path of G
    :param chunk_slice: chunk slice
    :param sources: sources array
    :param chunksize: size of chunk
    :param N: whole data size
    :return: 0
    """
    chunk_G = numpy.zeros((chunksize, chunksize), dtype=complex)
    for x, y, i in sources:
        # calulate chunk DFT
        u_chunk, v_chunk = (
            numpy.mgrid[
                -N // 2
                + chunk_slice[0].start : N // 2
                - (N - chunk_slice[0].stop),
                -N // 2
                + chunk_slice[1].start : N // 2
                - (N - chunk_slice[1].stop),
            ][::-1]
            / N
        )
        chunk_G += i * numpy.exp(2j * numpy.pi * (x * u_chunk + y * v_chunk))

    # lock
    lock = Lock(G_2_path)
    lock.acquire()
    with h5py.File(G_2_path, "r+") as f:
        dataset = f["G_data"]
        dataset[chunk_slice[0], chunk_slice[1]] = chunk_G / (N * N)
    lock.release()
    return 0


def make_G_2_FG_2_hdf5(
    sparse_ft_class, G_2_path, FG_2_path, chunksize, client
):
    """Generate standard data G and FG
        with hdf5

    :param sparse_ft_class: sparse_ft_class
    :param G_2_path: the hdf5 file path of G
    :param FG_2_path: the hdf5 file path of FG
    :param chunksize: size of chunk
    :param client: dask client

    :returns: G,FG
    """
    if chunksize is None:
        chunksize = sparse_ft_class.N // 8
    if not os.path.exists(FG_2_path):
        source_count = 1000
        sources = numpy.array(
            [
                (
                    numpy.random.randint(
                        -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                    ),
                    numpy.random.randint(
                        -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                    ),
                    numpy.random.rand()
                    * sparse_ft_class.N
                    * sparse_ft_class.N
                    / numpy.sqrt(source_count)
                    / 2,
                )
                for _ in range(source_count)
            ]
        )
        f = h5py.File(FG_2_path, "w")
        FG_dataset = f.create_dataset(
            "FG_data",
            (sparse_ft_class.N, sparse_ft_class.N),
            dtype="complex128",
        )
        # write data point by point
        for x, y, i in sources:
            FG_dataset[
                int(y) + sparse_ft_class.N // 2,
                int(x) + sparse_ft_class.N // 2,
            ] += i
        f.close()

    if not os.path.exists(G_2_path):
        # create a empty hdf5 file
        f = h5py.File(G_2_path, "w")
        G_dataset = f.create_dataset(
            "G_data",
            (sparse_ft_class.N, sparse_ft_class.N),
            dtype="complex128",
            chunks=(chunksize, chunksize),
        )
        chunk_list = []
        for chunk_slice in G_dataset.iter_chunks():
            chunk_list.append(
                DFT_chunk_work(
                    G_2_path,
                    chunk_slice,
                    sources,
                    chunksize,
                    sparse_ft_class.N,
                    use_dask=True,
                    nout=1,
                )
            )
        f.close()

        # compute
        chunk_list = client.compute(chunk_list, sync=True)

    return G_2_path, FG_2_path


@dask_wrapper
def error_task_subgrid_to_facet_2d(approx, true_image, num_true, **kwargs):
    """
    Calculate the error terms for the 2D subgrid to facet algorithm.

    :param approx: array of individual subgrid contributions
    :param constants_class: BaseArrays or SparseFourierTransform class object
                            containing fundamental and derived parameters
    :param facet_2: 2D numpy array of facets
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    """
    approx_img = numpy.zeros_like(approx) + approx
    err_mean = (
        numpy.abs(ifft(ifft(approx_img - true_image, axis=0), axis=1)) ** 2
        / num_true**2
    )
    err_mean_img = numpy.abs(approx_img - true_image) ** 2 / num_true**2
    return err_mean, err_mean_img


@dask_wrapper
def error_task_facet_to_subgrid_2d(approx, true_image, num_true, **kwargs):
    approx_img = numpy.zeros_like(approx) + approx
    err_mean = numpy.abs(approx_img - true_image) ** 2 / num_true**2
    err_mean_img = numpy.abs(
        fft(fft(approx_img - true_image, axis=0), axis=1) ** 2 / num_true**2
    )
    return err_mean, err_mean_img


# TODO: need reduce sum
@dask_wrapper
def sum_error_task(err_list, **kwargs):
    err_mean = numpy.zeros_like(err_list[0][0])
    err_mean_img = numpy.zeros_like(err_list[0][1])

    for err, err_img in err_list:
        err_mean += err
        err_mean_img += err_img
    return err_mean, err_mean_img


@dask_wrapper
def mean_img_task(err_img, **kwargs):
    return numpy.sqrt(numpy.mean(err_img[0])), numpy.sqrt(
        numpy.mean(err_img[1])
    )


def fund_errors(approx_what, number_what, true_what, error_task):
    error_task_list = []
    for i0, i1 in itertools.product(range(number_what), range(number_what)):
        tmp_error = error_task(
            approx_what[i0][i1],
            true_what[i0][i1],
            number_what,
            use_dask=True,
            nout=1,
        )
        error_task_list.append(tmp_error)

    error_sum_map = sum_error_task(error_task_list, use_dask=True, nout=1)
    mean_number = mean_img_task(error_sum_map, use_dask=True, nout=1)
    return mean_number


def errors_facet_to_subgrid_2d_dask(
    approx_subgrid, sparse_ft_class, subgrid_2, to_plot, fig_name
):
    return fund_errors(
        approx_subgrid,
        sparse_ft_class.nsubgrid,
        subgrid_2,
        error_task_facet_to_subgrid_2d,
    )


def errors_subgrid_to_facet_2d_dask(
    approx_facet, facet_2, sparse_ft_class, to_plot, fig_name
):
    return fund_errors(
        approx_facet,
        sparse_ft_class.nfacet,
        facet_2,
        error_task_subgrid_to_facet_2d,
    )


@dask_wrapper
def single_write_hdf5_task(
    hdf5_path,
    dataset_name,
    N,
    offset_i,
    block_size,
    base_arrays,
    idx0,
    idx1,
    block_data,
    **kwargs,
):
    if dataset_name == "G_data":
        mask_element_in = base_arrays.subgrid_A
    elif dataset_name == "FG_data":
        mask_element_in = base_arrays.facet_B
    else:
        raise ValueError("unsupport dataset_name")
    mask_element = numpy.outer(
        mask_element_in[idx0],
        mask_element_in[idx1],
    )
    block_data = block_data * mask_element
    slicex, slicey = roll_and_extract_mid(
        N, -offset_i[0], block_size
    ), roll_and_extract_mid(N, -offset_i[1], block_size)

    if len(slicex) <= len(slicey):
        iter_what1 = slicex
        iter_what2 = slicey
    else:
        iter_what1 = slicey
        iter_what2 = slicex

    pointx = [0]
    for sl in slicex:
        dt = sl.stop - sl.start
        pointx.append(dt + pointx[-1])

    pointy = [0]
    for sl in slicey:
        dt = sl.stop - sl.start
        pointy.append(dt + pointy[-1])

    # write with Lock
    lock = Lock(hdf5_path)
    lock.acquire()
    f = h5py.File(hdf5_path, "r+")
    approx_image_dataset = f[dataset_name]

    for i0 in range(len(iter_what1)):
        for i1 in range(len(iter_what2)):
            if len(slicex) <= len(slicey):
                sltestx = slice(pointx[i0], pointx[i0 + 1])
                sltesty = slice(pointy[i1], pointy[i1 + 1])
                approx_image_dataset[slicex[i0], slicey[i1]] += block_data[
                    sltestx, sltesty
                ]  # write it
            else:
                sltestx = slice(pointx[i1], pointx[i1 + 1])
                sltesty = slice(pointy[i0], pointy[i0 + 1])
                approx_image_dataset[slicex[i1], slicey[i0]] += block_data[
                    sltestx, sltesty
                ]

    f.close()
    lock.release()
    return hdf5_path


@dask_wrapper
def trim(ls, **kwargs):
    return ls[0]


def write_hdf5(
    approx_subgrid,
    approx_facet,
    approx_subgrid_path,
    approx_facet_path,
    sparse_ft_class,
    base_arrays_submit,
):

    # subgrid
    f = h5py.File(approx_subgrid_path, "w")
    G_dataset = f.create_dataset(
        "G_data", (sparse_ft_class.N, sparse_ft_class.N), dtype="complex128"
    )
    f.close()
    subgrid_res_list = []
    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        res = single_write_hdf5_task(
            approx_subgrid_path,
            "G_data",
            sparse_ft_class.N,
            (
                -sparse_ft_class.subgrid_off[i0],
                -sparse_ft_class.subgrid_off[i1],
            ),
            sparse_ft_class.xA_size,
            base_arrays_submit,
            i0,
            i1,
            approx_subgrid[i0][i1],
            use_dask=True,
            nout=1,
        )
        subgrid_res_list.append(res)

    # facets
    f = h5py.File(approx_facet_path, "w")
    FG_dataset = f.create_dataset(
        "FG_data", (sparse_ft_class.N, sparse_ft_class.N), dtype="complex128"
    )
    f.close()
    facet_res_list = []
    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        res = single_write_hdf5_task(
            approx_facet_path,
            "FG_data",
            sparse_ft_class.N,
            (
                -sparse_ft_class.facet_off[j0],
                -sparse_ft_class.facet_off[j1],
            ),
            sparse_ft_class.yB_size,
            base_arrays_submit,
            j0,
            j1,
            approx_facet[j0][j1],
            use_dask=True,
            nout=1,
        )
        facet_res_list.append(res)

    return trim(subgrid_res_list, use_dask=True, nout=1), trim(
        facet_res_list, use_dask=True, nout=1
    )
