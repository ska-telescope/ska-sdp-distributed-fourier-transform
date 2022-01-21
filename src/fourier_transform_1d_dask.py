#!/usr/bin/env python
# coding: utf-8
import logging
import math
import numpy
import sys

from matplotlib import pylab

from src.fourier_transform.fourier_algorithm import (
    make_subgrid_and_facet,
    facets_to_subgrid_1,
    facets_to_subgrid_2,
    subgrid_to_facet_1,
    subgrid_to_facet_2,
    get_actual_work_terms,
    calculate_pswf,
    generate_mask_for_subgrid_facet,
)
from src.fourier_transform.utils import (
    whole,
    plot_1,
    plot_2,
    plot_errors_subgrid_1D,
    plot_errors_facet_1D,
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
    "W": 13.25,
    "fov": 0.75,
    "N": 1024,  # total image size
    "Nx": 4,  # ??
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


def main():
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
    xM_yP_size = xM_size * yP_size // N
    xMxN_yP_size = xM_yP_size + int(2 * numpy.ceil(xN_size * yP_size / N / 2))
    assert (xM_size * yN_size) % N == 0
    xM_yN_size = xM_size * yN_size // N

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

    plot_1(pswf, xN, xN_size, yB, yN, N, yN_size)

    # Calculate actual work terms to use. We need both $n$ and $b$ in image space.
    Fb, Fn, facet_m0_trunc = get_actual_work_terms(
        pswf, xM, xMxN_yP_size, yB_size, yN_size, xM_size, N, yP_size
    )
    plot_2(facet_m0_trunc, xM, xMxN_yP_size, yP_size)

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

    facet_B, subgrid_A = generate_mask_for_subgrid_facet(
        facet_off, nfacet, nsubgrid, subgrid_off, xA_size, N, yB_size
    )

    G = numpy.random.rand(N) - 0.5
    subgrid, facet = make_subgrid_and_facet(
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

    log.info("\n== RUN: Facet to subgrid")
    # With a few more slight optimisations we arrive at a compact representation for our algorithm.
    # For reference, what we are computing here is:

    dtype = numpy.complex128
    xN_yP_size = xMxN_yP_size - xM_yP_size

    log.info("Facet data: %s %s", facet.shape, facet.size)
    nmbfs = facets_to_subgrid_1(
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
    approx_subgrid = numpy.array(
        [
            facets_to_subgrid_2(
                nmbfs, i, xM_size, nfacet, facet_off, N, subgrid_A, xA_size
            )
            for i in range(nsubgrid)
        ]
    )
    log.info("Reconstructed subgrids: %s %s", approx_subgrid.shape, approx_subgrid.size)

    plot_errors_subgrid_1D(approx_subgrid, nsubgrid, subgrid, xA, xA_size, N)

    #  By feeding the implementation single-pixel inputs we can create a full error map.

    log.info("\n== RUN: Subgrid to facet")
    log.info("Subgrid data: %s %s", subgrid.shape, subgrid.size)
    nafs = subgrid_to_facet_1(
        subgrid, nsubgrid, nfacet, xM_yN_size, xM_size, facet_off, N, Fn
    )

    # - redistribution of FNjSi here -
    log.info("Intermediate data: %s %s", nafs.shape, nafs.size)
    approx_facet = numpy.array(
        [
            subgrid_to_facet_2(
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
            )
            for j in range(nfacet)
        ]
    )
    log.info("Reconstructed facets: %s %s", approx_facet.shape, approx_facet.size)

    plot_errors_facet_1D(approx_facet, facet, nfacet, xA, xM, yB, yB_size)


if __name__ == "__main__":
    main()
