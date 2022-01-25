#!/usr/bin/env python
# coding: utf-8
import itertools
import logging
import math
import time
import numpy
import sys

from matplotlib import pylab

from src.fourier_transform.fourier_algorithm import (
    ifft,
    extract_mid,
    fft,
    prepare_facet,
    extract_subgrid,
    get_actual_work_terms,
    calculate_pswf,
    generate_mask,
)
from src.fourier_transform.utils import (
    whole,
    plot_1,
    plot_2,
    calculate_and_plot_errors_2d,
    test_accuracy_facet_to_subgrid,
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

TARGET_ERR = 1e-5
ALPHA = 0


def main(to_plot=True, fig_name=None):
    """
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix into PNG files.
                     If to_plot is set to False, fig_name doesn't have an effect.
    """
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

    # ## 2D case
    #
    # All of this generalises to two dimensions in the way you would expect. Let us set up test data:

    log.info("%s x %s subgrids %s x %s facets", nsubgrid, nsubgrid, nfacet, nfacet)
    subgrid_2 = numpy.empty((nsubgrid, nsubgrid, xA_size, xA_size), dtype=complex)
    facet_2 = numpy.empty((nfacet, nfacet, yB_size, yB_size), dtype=complex)

    # adding sources
    add_sources = True
    if add_sources:
        FG_2 = numpy.zeros((N, N))
        source_count = 1000
        sources = [
            (
                numpy.random.randint(-N // 2, N // 2 - 1),
                numpy.random.randint(-N // 2, N // 2 - 1),
                numpy.random.rand() * N * N / numpy.sqrt(source_count) / 2,
            )
            for _ in range(source_count)
        ]
        for x, y, i in sources:
            FG_2[y + N // 2, x + N // 2] += i
        G_2 = ifft(FG_2)

    else:
        # without sources
        G_2 = (
            numpy.exp(2j * numpy.pi * numpy.random.rand(N, N))
            * numpy.random.rand(N, N)
            / 2
        )
        FG_2 = fft(G_2)

    log.info("Mean grid absolute: %s", numpy.mean(numpy.abs(G_2)))

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

    # RUN FACET -> SUBGRID ALGORITHM

    # 3 Approaches:
    #   - they differ in when facet is prepared and which axis is run first
    #   - they all give the same result, but with a different speed
    #   - #1 is slowest, because that prepares all facets first, which substantially increases their size
    #     and hence, puts a large amount of data into the following loops

    # Approach 1: do prepare_facet step across both axes first, then go into the loop
    #             over subgrids horizontally (axis=0) and within that, loop over subgrids
    #             vertically (axis=1) and do the extract_subgrid step in these two directions
    # Gabi ^ Peter v
    # Having those operations separately means that we can shuffle things around quite a bit
    # without affecting the result. The obvious first choice might be to do all facet-preparation
    # up-front, as this allows us to share the computation across all subgrids:
    t = time.time()
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
    log.info("%s s", time.time() - t)

    # Approach 2: First, do prepare_facet on the horizontal axis (axis=0), then
    #             for loop for the horizontal subgrid direction, and do extract_subgrid
    #             within same loop do prepare_facet in the vertical case (axis=1), then
    #             go into the fila subgrid loop in the vertical dir and do extract_subgrid for that
    # Gabi ^ Peter v
    # # However, remember that `prepare_facet` increases the amount of data involved, which in turn
    # means that we need to shuffle more data through subsequent computations.
    # #
    # # Therefore it is actually more efficient to first do the subgrid-specific reduction, and *then*
    # continue with the (constant) facet preparation along the other axis. We can tackle both axes in
    # whatever order we like, it doesn't make a difference for the result:
    t = time.time()
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        BF_F = prepare_facet(facet_2[j0, j1], 0, Fb, yP_size)
        for i0 in range(nsubgrid):
            NMBF_F = extract_subgrid(
                BF_F,
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
            NMBF_BF = prepare_facet(NMBF_F, 1, Fb, yP_size)
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
    log.info("%s s", time.time() - t)

    # Approach 3: same as 2, but starts with the vertical direction (axis=1)
    #             and finishes with the horizontal (axis=0) axis
    # Gabi ^
    t = time.time()
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        F_BF = prepare_facet(facet_2[j0, j1], 1, Fb, yP_size)
        for i1 in range(nsubgrid):
            F_NMBF = extract_subgrid(
                F_BF,
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
            BF_NMBF = prepare_facet(F_NMBF, 0, Fb, yP_size)
            for i0 in range(nsubgrid):
                NMBF_NMBF[i0, i1, j0, j1] = extract_subgrid(
                    BF_NMBF,
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
    log.info("%s s", time.time() - t)

    calculate_and_plot_errors_2d(
        NMBF_NMBF,
        facet_off,
        nfacet,
        nsubgrid,
        subgrid_2,
        subgrid_A,
        xM_size,
        N,
        xA_size,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    test_accuracy_facet_to_subgrid(
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
        to_plot=to_plot,
        fig_name=fig_name,
    )


if __name__ == "__main__":
    main()


# TODO: add the subgrid-to-facet 2d bit
#  --> created by Celeste; code reference is at the end of test_algorithm.py
