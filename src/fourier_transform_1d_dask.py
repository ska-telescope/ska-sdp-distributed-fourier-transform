#!/usr/bin/env python
# coding: utf-8
import logging
import math
import os

import numpy
import sys
import dask

from distributed import performance_report
from matplotlib import pylab

from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask

from src.fourier_transform.fourier_algorithm import (
    fft,
    make_subgrid_and_facet,
    facets_to_subgrid_1d,
    reconstruct_subgrid_1d,
    subgrid_to_facet_1d,
    reconstruct_facet_1d,
    get_actual_work_terms,
    calculate_pswf,
    generate_mask,
    subgrid_to_facet_1d_dask_array,
    facets_to_subgrid_1d_dask_array,
    make_subgrid_and_facet_dask_array,
    reconstruct_subgrid_1d_dask_array,
    reconstruct_facet_1d_dask_array,
)
from src.fourier_transform.utils import (
    whole,
    plot_1,
    plot_2,
    calculate_and_plot_errors_subgrid_1d,
    calculate_and_plot_errors_facet_1d,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Fixing seed of numpy random
numpy.random.seed(123456789)

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

# expand these, instead of running exec(f"{n} = target_pars[n]") to fix code syntax
W = TARGET_PARS["W"]
fov = TARGET_PARS["fov"]
N = TARGET_PARS["N"]
Nx = TARGET_PARS["Nx"]
yB_size = TARGET_PARS["yB_size"]
yN_size = TARGET_PARS["yN_size"]
yP_size = TARGET_PARS["yP_size"]
xA_size = TARGET_PARS["xA_size"]
xM_size = TARGET_PARS["xM_size"]

ALPHA = 0


def _algorithm_with_dask_array(
    G,
    nsubgrid,
    nfacet,
    subgrid_A,
    facet_B,
    subgrid_off,
    facet_off,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    xM_yN_size,
    Fb,
    Fn,
    facet_m0_trunc,
    dtype,
):
    subgrid, facet = make_subgrid_and_facet_dask_array(
        G,
        nsubgrid,
        xA_size,
        subgrid_A,
        subgrid_off,
        nfacet,
        yB_size,
        facet_B,
        facet_off,
    )

    # ==============================================
    log.info("\n== RUN: Facet to subgrid")
    # With a few more slight optimisations we arrive at a compact representation for our algorithm.
    # For reference, what we are computing here is:
    log.info("Facet data: %s %s", facet.shape, facet.size)

    nmbfs = facets_to_subgrid_1d_dask_array(
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
    )
    # - redistribution of nmbfs here -
    log.info("Redistributed data: %s %s", nmbfs.shape, nmbfs.size)

    approx_subgrid = reconstruct_subgrid_1d_dask_array(
        nmbfs, xM_size, nfacet, facet_off, N, subgrid_A, xA_size, nsubgrid
    )
    log.info("Reconstructed subgrids: %s %s", approx_subgrid.shape, approx_subgrid.size)

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")
    log.info("Subgrid data: %s %s", subgrid.shape, subgrid.size)

    nafs = subgrid_to_facet_1d_dask_array(
        subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
    )
    # - redistribution of FNjSi here -
    log.info("Intermediate data: %s %s", nafs.shape, nafs.size)

    approx_facet = reconstruct_facet_1d_dask_array(
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
    )
    log.info("Reconstructed facets: %s %s", approx_facet.shape, approx_facet.size)

    return subgrid, facet, approx_subgrid, approx_facet


def _algorithm_with_dask_delayed(
    G,
    nsubgrid,
    nfacet,
    subgrid_A,
    facet_B,
    subgrid_off,
    facet_off,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    xM_yN_size,
    Fb,
    Fn,
    facet_m0_trunc,
    dtype,
):
    FG = fft(G)
    subgrid, facet = make_subgrid_and_facet(
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
        1,
        use_dask=True,
    )
    # ==============================================
    log.info("\n== RUN: Facet to subgrid")

    nmbfs = facets_to_subgrid_1d(
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
        use_dask=True,
    )

    approx_subgrid = reconstruct_subgrid_1d(
        nmbfs,
        xM_size,
        nfacet,
        facet_off,
        N,
        subgrid_A,
        xA_size,
        nsubgrid,
        use_dask=True,
    )

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")

    nafs = subgrid_to_facet_1d(
        subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn, use_dask=True
    )

    approx_facet = reconstruct_facet_1d(
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
    nsubgrid,
    nfacet,
    subgrid_A,
    facet_B,
    subgrid_off,
    facet_off,
    xMxN_yP_size,
    xN_yP_size,
    xM_yP_size,
    xM_yN_size,
    Fb,
    Fn,
    facet_m0_trunc,
    dtype,
):
    FG = fft(G)
    subgrid, facet = make_subgrid_and_facet(
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
        use_dask=False,
    )
    # - redistribution of nmbfs here -
    log.info("Redistributed data: %s %s", nmbfs.shape, nmbfs.size)

    approx_subgrid = reconstruct_subgrid_1d(
        nmbfs,
        xM_size,
        nfacet,
        facet_off,
        N,
        subgrid_A,
        xA_size,
        nsubgrid,
        use_dask=False,
    )
    log.info("Reconstructed subgrids: %s %s", approx_subgrid.shape, approx_subgrid.size)

    # ==============================================
    log.info("\n== RUN: Subgrid to facet")
    log.info("Subgrid data: %s %s", subgrid.shape, subgrid.size)

    nafs = subgrid_to_facet_1d(
        subgrid,
        nsubgrid,
        nfacet,
        xM_yN_size,
        xM_size,
        facet_off,
        N,
        Fn,
        use_dask=False,
    )
    # - redistribution of FNjSi here -
    log.info("Intermediate data: %s %s", nafs.shape, nafs.size)

    approx_facet = reconstruct_facet_1d(
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

    use_dask = os.getenv("USE_DASK", False) == "True"

    log.info("== Chosen configuration")
    for n in [
        "W",
        "fov",
        "N",
        "Nx",
        "yB_size",
        "yN_size",
        "yP_size",
        "xA_size",
        "xM_size",
    ]:
        log.info(f"{n} = {TARGET_PARS[n]}")

    log.info("\n== Relative coordinates")
    xN = W / yN_size / 2
    xM = xM_size / 2 / N
    yN = yN_size / 2
    xA = xA_size / 2 / N
    yB = yB_size / 2
    log.info("xN=%g xM=%g yN=%g xNyN=%g xA=%g" % (xN, xM, yN, xN * yN, xA))

    log.info("\n== Derived values")
    xN_size = N * W / yN_size

    # Note from Peter: Note that this (xM_yP_size) is xM_yN_size * yP_size / yN_size.
    # So could replace by yN_size, which would be the more fundamental entity.
    # Generally - if you don't want to carry a structure around with all the derived values,
    # it would probably best just to re-derive these entities from the fundamental sizes when required.
    # I mean, the names basically tell you how to calculate them - x[a]_y[b]_size = x[a]_size * y[b]_size / N
    # ^ Gabi: we will have to deal with the input params and derived values better,
    #         this could be a way of refactoring + keeping them in a class that does the necessary calcs
    xM_yP_size = xM_size * yP_size // N

    xMxN_yP_size = xM_yP_size + int(2 * numpy.ceil(xN_size * yP_size / N / 2))
    assert (xM_size * yN_size) % N == 0

    # same not as above xM_yP_size; N could be replaced xM_size
    xM_yN_size = xM_size * yN_size // N

    xN_yP_size = xMxN_yP_size - xM_yP_size

    log.info(
        f"xN_size={xN_size:.1f} xM_yP_size={xM_yP_size}, xMxN_yP_size={xMxN_yP_size}, xM_yN_size={xM_yN_size}"
    )

    if fov is not None:
        nfacet = int(numpy.ceil(N * fov / yB_size))
        log.info(
            f"{nfacet}x{nfacet} facets for FoV of {fov} ({N * fov / nfacet / yB_size * 100}% efficiency)"
        )

    log.info("\n== Calculate PSWF")
    pswf = calculate_pswf(yN_size, ALPHA, W)

    if to_plot:
        plot_1(pswf, xN, xN_size, yB, yN, N, yN_size, fig_name=fig_name)

    # Calculate actual work terms to use. We need both $n$ and $b$ in image space.
    Fb, Fn, facet_m0_trunc = get_actual_work_terms(
        pswf, xM, xMxN_yP_size, yB_size, yN_size, xM_size, N, yP_size
    )

    if to_plot:
        plot_2(facet_m0_trunc, xM, xMxN_yP_size, yP_size, fig_name=fig_name)

    log.info("\n== Generate layout (factes and subgrids")
    # Layout subgrids + facets
    nsubgrid = int(math.ceil(N / xA_size))
    nfacet = int(math.ceil(N / yB_size))
    log.info("%d subgrids, %d facets needed to cover" % (nsubgrid, nfacet))
    subgrid_off = xA_size * numpy.arange(nsubgrid) + Nx
    facet_off = yB_size * numpy.arange(nfacet)

    assert whole(numpy.outer(subgrid_off, facet_off) / N)
    assert whole(facet_off * xM_size / N)

    log.info("\n== Generate A/B masks and subgrid/facet offsets")
    # Determine subgrid/facet offsets and the appropriate A/B masks for cutting them out.
    # We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

    facet_B = generate_mask(N, nfacet, yB_size, facet_off)
    subgrid_A = generate_mask(N, nsubgrid, xA_size, subgrid_off)

    G = numpy.random.rand(N) - 0.5
    dtype = numpy.complex128

    if use_dask and dask_option == "array":
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_with_dask_array(
            G,
            nsubgrid,
            nfacet,
            subgrid_A,
            facet_B,
            subgrid_off,
            facet_off,
            xMxN_yP_size,
            xN_yP_size,
            xM_yP_size,
            xM_yN_size,
            Fb,
            Fn,
            facet_m0_trunc,
            dtype,
        )
        subgrid, facet, approx_subgrid, approx_facet = dask.compute(
            subgrid, facet, approx_subgrid, approx_facet
        )

    elif use_dask and dask_option == "delayed":
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_with_dask_delayed(
            G,
            nsubgrid,
            nfacet,
            subgrid_A,
            facet_B,
            subgrid_off,
            facet_off,
            xMxN_yP_size,
            xN_yP_size,
            xM_yP_size,
            xM_yN_size,
            Fb,
            Fn,
            facet_m0_trunc,
            dtype,
        )

    else:
        subgrid, facet, approx_subgrid, approx_facet = _algorithm_in_serial(
            G,
            nsubgrid,
            nfacet,
            subgrid_A,
            facet_B,
            subgrid_off,
            facet_off,
            xMxN_yP_size,
            xN_yP_size,
            xM_yP_size,
            xM_yN_size,
            Fb,
            Fn,
            facet_m0_trunc,
            dtype,
        )

    calculate_and_plot_errors_subgrid_1d(
        approx_subgrid,
        nsubgrid,
        subgrid,
        xA,
        xA_size,
        N,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    calculate_and_plot_errors_facet_1d(
        approx_facet,
        facet,
        nfacet,
        xA,
        xM,
        yB,
        yB_size,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    return subgrid, facet, approx_subgrid, approx_facet


if __name__ == "__main__":

    client, current_env_var = set_up_dask()
    with performance_report(filename="dask-report.html"):
        main(to_plot=False)
    tear_down_dask(client, current_env_var)

    # all above needs commenting and this uncommenting if want to run it without dask
    # main(to_plot=False)
