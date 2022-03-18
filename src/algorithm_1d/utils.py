import logging
import numpy
from matplotlib import pylab

from src.fourier_transform.fourier_algorithm import coordinates, fft, ifft
from src.utils import mark_range

log = logging.getLogger("fourier-logger")


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
            ax2.semilogy(N * coordinates(xA_size), numpy.abs(fft(error, axis=0)))
        err_sum += numpy.abs(error) ** 2 / nsubgrid
        err_sum_img += numpy.abs(fft(error, axis=0)) ** 2 / nsubgrid

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
        err_sum += numpy.abs(ifft(error, axis=0)) ** 2
        err_sum_img += numpy.abs(error) ** 2
        if to_plot:
            ax1.semilogy(
                coordinates(constants_class.yB_size),
                numpy.abs(ifft(error, axis=0)),
            )
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
