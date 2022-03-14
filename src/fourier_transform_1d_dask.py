#!/usr/bin/env python
# coding: utf-8
import logging

import numpy
import sys
import dask

from distributed import performance_report
from matplotlib import pylab

from src.fourier_transform.algorithm_parameters import DistributedFFT
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask

from src.fourier_transform.fourier_algorithm import (
    fft_along_axis,
    make_subgrid_and_facet,
    facets_to_subgrid_1d,
    reconstruct_subgrid_1d,
    subgrid_to_facet_1d,
    reconstruct_facet_1d,
    subgrid_to_facet_1d_dask_array,
    facets_to_subgrid_1d_dask_array,
    make_subgrid_and_facet_dask_array,
    reconstruct_subgrid_1d_dask_array,
    reconstruct_facet_1d_dask_array,
)
from src.fourier_transform.utils import (
    plot_1,
    plot_2,
    calculate_and_plot_errors_subgrid_1d,
    calculate_and_plot_errors_facet_1d,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Plot setup
pylab.rcParams["figure.figsize"] = 16, 4
pylab.rcParams["image.cmap"] = "viridis"

# A / x --> grid (frequency) space; B / y --> image (facet) space

# 65K params for large-scale testing (copied from Peter's Jupyter notebook)
# TARGET_PARS = {
#     "W": 13.65625,  # 13.25,
#     "fov": 0.75,
#     "N": 65536,  # 1024,  # total image size
#     "Nx": 256,  # 4,  # ??
#     "yB_size": 6144,  # 256,  # true usable image size (facet)
#     "yN_size": 7552,  # 320,  # padding needed to transfer the data?
#     "yP_size": 16384,  # 512,  # padded (rough) image size (facet)
#     "xA_size": 256,  # 188,  # true usable subgrid size
#     "xM_size": 512,  # 256,  # padded (rough) subgrid size
# }

TARGET_PARS = {
    "W": 13.25,  # PSWF parameter (grid-space support)
    "fov": 0.75,
    "N": 1024,  # total image size
    "Nx": 4,  # subgrid spacing: it tells you what subgrid offsets are permissible:
    # here it is saying that they need to be divisible by 4.
    "yB_size": 256,  # true usable image size (facet)
    "yN_size": 320,  # padding needed to transfer the data?
    "yP_size": 512,  # padded (rough) image size (facet)
    "xA_size": 188,  # true usable subgrid size
    "xM_size": 256,  # padded (rough) subgrid size
}

ALPHA = 0


def _algorithm_with_dask_array(
    G,
    distr_fft_class,
    dtype,
):
    FG = fft_along_axis(G, axis=0)
    subgrid, facet = make_subgrid_and_facet_dask_array(
        G,
        FG,
        distr_fft_class,
    )

    # ==============================================
    log.info("\n== RUN: Facet to subgrid")
    # With a few more slight optimisations we arrive at a compact representation for our algorithm.
    # For reference, what we are computing here is:
    log.info("Facet data: %s %s", facet.shape, facet.size)

    nmbfs = facets_to_subgrid_1d_dask_array(
        facet,
        distr_fft_class,
        dtype,
    )
    # - redistribution of nmbfs here -
    log.info("Redistributed data: %s %s", nmbfs.shape, nmbfs.size)

    approx_subgrid = reconstruct_subgrid_1d_dask_array(nmbfs, distr_fft_class)
    log.info("Reconstructed subgrids: %s %s", approx_subgrid.shape, approx_subgrid.size)

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")
    log.info("Subgrid data: %s %s", subgrid.shape, subgrid.size)

    nafs = subgrid_to_facet_1d_dask_array(
        subgrid,
        distr_fft_class,
    )
    # - redistribution of FNjSi here -
    log.info("Intermediate data: %s %s", nafs.shape, nafs.size)

    approx_facet = reconstruct_facet_1d_dask_array(
        nafs,
        distr_fft_class,
    )
    log.info("Reconstructed facets: %s %s", approx_facet.shape, approx_facet.size)

    return subgrid, facet, approx_subgrid, approx_facet


def _algorithm_with_dask_delayed(
    G,
    distr_fft_class,
    dtype,
):
    FG = fft_along_axis(G, axis=0)
    subgrid, facet = make_subgrid_and_facet(
        G,
        FG,
        distr_fft_class,
        1,
        use_dask=True,
    )
    # ==============================================
    log.info("\n== RUN: Facet to subgrid")

    nmbfs = facets_to_subgrid_1d(
        facet,
        distr_fft_class,
        dtype,
        use_dask=True,
    )

    approx_subgrid = reconstruct_subgrid_1d(
        nmbfs,
        distr_fft_class,
        use_dask=True,
    )

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")

    nafs = subgrid_to_facet_1d(subgrid, distr_fft_class, use_dask=True)

    approx_facet = reconstruct_facet_1d(
        nafs,
        distr_fft_class,
        use_dask=True,
    )

    subgrid, facet, approx_subgrid, approx_facet = dask.compute(
        subgrid, facet, approx_subgrid, approx_facet
    )

    subgrid = numpy.array(subgrid)
    facet = numpy.array(facet)
    approx_subgrid = numpy.array(approx_subgrid)
    approx_facet = numpy.array(approx_facet)

    return subgrid, facet, approx_subgrid, approx_facet


def _algorithm_in_serial(
    G,
    distr_fft_class,
    dtype,
):
    FG = fft_along_axis(G, axis=0)
    subgrid, facet = make_subgrid_and_facet(
        G,
        FG,
        distr_fft_class,
        dims=1,
        use_dask=False,
    )

    # ==============================================
    log.info("\n== RUN: Facet to subgrid")
    # With a few more slight optimisations we arrive at a compact representation for our algorithm.
    # For reference, what we are computing here is:
    log.info("Facet data: %s %s", facet.shape, facet.size)

    nmbfs = facets_to_subgrid_1d(
        facet,
        distr_fft_class,
        dtype,
        use_dask=False,
    )
    # - redistribution of nmbfs here -
    log.info("Redistributed data: %s %s", nmbfs.shape, nmbfs.size)

    approx_subgrid = reconstruct_subgrid_1d(
        nmbfs,
        distr_fft_class,
        use_dask=False,
    )
    log.info("Reconstructed subgrids: %s %s", approx_subgrid.shape, approx_subgrid.size)

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")
    log.info("Subgrid data: %s %s", subgrid.shape, subgrid.size)

    nafs = subgrid_to_facet_1d(
        subgrid,
        distr_fft_class,
        use_dask=False,
    )
    # - redistribution of FNjSi here -
    log.info("Intermediate data: %s %s", nafs.shape, nafs.size)

    approx_facet = reconstruct_facet_1d(
        nafs,
        distr_fft_class,
        use_dask=False,
    )
    log.info("Reconstructed facets: %s %s", approx_facet.shape, approx_facet.size)

    return subgrid, facet, approx_subgrid, approx_facet


def main(to_plot=True, fig_name=None, use_dask=False, dask_option="array"):
    """
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix into PNG files.
                     If to_plot is set to False, fig_name doesn't have an effect.
    :param use_dask: use dask?
    :param dask_option: Dask optimisation option -- array or delayed
    """
    log.info("== Chosen configuration")
    distr_fft_class = DistributedFFT(**TARGET_PARS)
    log.info(distr_fft_class)

    if to_plot:
        plot_1(distr_fft_class.pswf, distr_fft_class, fig_name=fig_name)
        plot_2(distr_fft_class, fig_name=fig_name)

    log.info("\n== Generate layout (factes and subgrids")
    # Layout subgrids + facets
    log.info(
        "%d subgrids, %d facets needed to cover"
        % (distr_fft_class.nsubgrid, distr_fft_class.nfacet)
    )
    distr_fft_class.check_offsets()

    log.info("\n== Generate A/B masks and subgrid/facet offsets")
    # Determine subgrid/facet offsets and the appropriate A/B masks for cutting them out.
    # We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

    G = numpy.random.rand(distr_fft_class.N) - 0.5
    dtype = numpy.complex128

    if use_dask and dask_option == "array":
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_with_dask_array(
            G,
            distr_fft_class,
            dtype,
        )
        subgrid, facet, approx_subgrid, approx_facet = dask.compute(
            subgrid, facet, approx_subgrid, approx_facet
        )

    elif use_dask and dask_option == "delayed":
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_with_dask_delayed(
            G,
            distr_fft_class,
            dtype,
        )

    else:
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_in_serial(
            G,
            distr_fft_class,
            dtype,
        )

    calculate_and_plot_errors_subgrid_1d(
        approx_subgrid,
        distr_fft_class.nsubgrid,
        subgrid,
        distr_fft_class.xA_size,
        distr_fft_class.N,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    calculate_and_plot_errors_facet_1d(
        approx_facet,
        facet,
        distr_fft_class,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    return subgrid, facet, approx_subgrid, approx_facet


if __name__ == "__main__":
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    client = set_up_dask()
    with performance_report(filename="dask-report.html"):
        main(to_plot=False, use_dask=True)
    tear_down_dask(client)

    # all above needs commenting and this uncommenting if want to run it without dask
    # main(to_plot=False)
