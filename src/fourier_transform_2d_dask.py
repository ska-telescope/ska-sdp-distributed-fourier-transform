#!/usr/bin/env python
# coding: utf-8
import itertools
import logging
import time

import dask
import numpy
import sys
import dask.array
from distributed import performance_report

from matplotlib import pylab

from src.fourier_transform.algorithm_parameters import DistributedFFT
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    ifft,
    fft,
    prepare_facet,
    extract_subgrid,
    prepare_subgrid,
    extract_facet_contribution,
    add_subgrid_contribution,
    finish_facet,
    make_subgrid_and_facet,
)
from src.fourier_transform.utils import (
    plot_1,
    plot_2,
    errors_facet_to_subgrid_2d,
    test_accuracy_facet_to_subgrid,
    test_accuracy_subgrid_to_facet,
    errors_subgrid_to_facet_2d,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Plot setup
pylab.rcParams["figure.figsize"] = 16, 4
pylab.rcParams["image.cmap"] = "viridis"

# A / x --> grid (frequency) space; B / y --> image (facet) space

TARGET_PARS = {
    "W": 13.25,  # PSWF parameter (grid-space support)
    "fov": 0.75,  # field of view?
    "N": 1024,  # total image size
    "Nx": 4,  # subgrid spacing: it tells you what subgrid offsets are permissible:
    # here it is saying that they need to be divisible by 4.
    "yB_size": 256,  # true usable image size (facet)
    "yN_size": 320,  # padding needed to transfer the data?
    "yP_size": 512,  # padded (rough) image size (facet)
    "xA_size": 188,  # true usable subgrid size
    "xM_size": 256,  # padded (rough) subgrid size
}

TARGET_ERR = 1e-5
ALPHA = 0


def _generate_naf_naf(
    subgrid_2, constants_class, use_dask
):
    naf_naf = numpy.empty(
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
    if use_dask:
        naf_naf = naf_naf.tolist()
    for i0, i1 in itertools.product(range(constants_class.nsubgrid), range(constants_class.nsubgrid)):
        AF_AF = prepare_subgrid(
            subgrid_2[i0][i1], constants_class.xM_size, use_dask=use_dask, nout=1
        )
        for j0 in range(constants_class.nfacet):
            NAF_AF = extract_facet_contribution(
                AF_AF,
                constants_class.facet_off[j0],
                constants_class,
                0,
                use_dask=use_dask,
                nout=1,
            )
            for j1 in range(constants_class.nfacet):
                naf_naf[i0][i1][j0][j1] = extract_facet_contribution(
                    NAF_AF,
                    constants_class.facet_off[j1],
                    constants_class,
                    1,
                    use_dask=use_dask,
                    nout=1,
                )
    return naf_naf


def subgrid_to_facet_algorithm(
    subgrid_2,
    constants_class,
    use_dask=False,
):
    naf_naf = _generate_naf_naf(
        subgrid_2,
        constants_class,
        use_dask,
    )

    BMNAF_BMNAF = numpy.empty(
        (constants_class.nfacet, constants_class.nfacet, constants_class.yB_size, constants_class.yB_size), dtype=complex
    )
    if use_dask:
        BMNAF_BMNAF = BMNAF_BMNAF.tolist()

    for j0, j1 in itertools.product(range(constants_class.nfacet), range(constants_class.nfacet)):
        MNAF_BMNAF = numpy.zeros(
            (constants_class.yP_size, constants_class.yB_size), dtype=complex
        )
        for i0 in range(constants_class.nsubgrid):
            NAF_MNAF = numpy.zeros(
                (constants_class.xM_yN_size, constants_class.yP_size), dtype=complex
            )
            for i1 in range(constants_class.nsubgrid):
                if use_dask:
                    NAF_MNAF = NAF_MNAF + dask.array.from_delayed(
                        add_subgrid_contribution(
                            len(NAF_MNAF.shape),
                            naf_naf[i0][i1][j0][j1],
                            constants_class.subgrid_off[i1],
                            constants_class,
                            1,
                            use_dask=use_dask,
                            nout=1,
                        ),
                        shape=(constants_class.xM_yN_size, constants_class.yP_size),
                        dtype=complex,
                    )
                else:
                    NAF_MNAF = NAF_MNAF + add_subgrid_contribution(
                        len(NAF_MNAF.shape),
                        naf_naf[i0][i1][j0][j1],
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
                use_dask=use_dask,
                nout=0,
            )
            if use_dask:
                MNAF_BMNAF = MNAF_BMNAF + dask.array.from_delayed(
                    add_subgrid_contribution(
                        len(MNAF_BMNAF.shape),
                        NAF_BMNAF,
                        constants_class.subgrid_off[i0],
                        constants_class,
                        0,
                        use_dask=use_dask,
                        nout=1,
                    ),
                    shape=(constants_class.yP_size, constants_class.yB_size),
                    dtype=complex,
                )
            else:
                MNAF_BMNAF = MNAF_BMNAF + add_subgrid_contribution(
                    len(MNAF_BMNAF.shape),
                    NAF_BMNAF,
                    constants_class.subgrid_off[i0],
                    constants_class,
                    0,
                    use_dask=use_dask,
                    nout=1,
                )
        BMNAF_BMNAF[j0][j1] = finish_facet(
            MNAF_BMNAF,
            constants_class.Fb,
            constants_class.facet_B[j0],
            constants_class.yB_size,
            0,
            use_dask=use_dask,
            nout=1,
        )

    return BMNAF_BMNAF


def facet_to_subgrid_2d_method_1(
    facet,
    constants_class,
    use_dask=False,
):
    """
    Generate subgrid from facet 2D. 1st Method.

    Approach 1: do prepare_facet step across both axes first, then go into the loop
                over subgrids horizontally (axis=0) and within that, loop over subgrids
                vertically (axis=1) and do the extract_subgrid step in these two directions

    Having those operations separately means that we can shuffle things around quite a bit
    without affecting the result. The obvious first choice might be to do all facet-preparation
    up-front, as this allows us to share the computation across all subgrids

    :param facet: 2D numpy array of facets
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: use dask.delayed or not

    :return: TODO ???
    """

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
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(range(constants_class.nfacet), range(constants_class.nfacet)):
        BF_F = prepare_facet(
            facet[j0][j1], 0, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
        )
        BF_BF = prepare_facet(
            BF_F, 1, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
        )
        for i0 in range(constants_class.nsubgrid):
            NMBF_BF = extract_subgrid(
                BF_BF,
                0,
                constants_class.subgrid_off[i0],
                constants_class,
                use_dask=use_dask,
                nout=1,
            )
            for i1 in range(constants_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][j1] = extract_subgrid(
                    NMBF_BF,
                    1,
                    constants_class.subgrid_off[i1],
                    constants_class,
                    use_dask=use_dask,
                    nout=1,
                )
    return NMBF_NMBF


def facet_to_subgrid_2d_method_2(
    NMBF_NMBF,
    facet,
    constants_class,
    use_dask=False,
):
    """
    Approach 2: First, do prepare_facet on the horizontal axis (axis=0), then
                for loop for the horizontal subgrid direction, and do extract_subgrid
                within same loop do prepare_facet in the vertical case (axis=1), then
                go into the fila subgrid loop in the vertical dir and do extract_subgrid for that

    However, remember that `prepare_facet` increases the amount of data involved, which in turn
    means that we need to shuffle more data through subsequent computations. Therefore it is actually
    more efficient to first do the subgrid-specific reduction, and *then* continue with the (constant)
    facet preparation along the other axis. We can tackle both axes in whatever order we like,
    it doesn't make a difference for the result.

    :param NMBF_NMBF: TODO ???
    :param facet: 2D numpy array of facets
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: use dask.delayed or not
    """
    for j0, j1 in itertools.product(range(constants_class.nfacet), range(constants_class.nfacet)):
        BF_F = prepare_facet(
            facet[j0][j1], 0, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
        )
        for i0 in range(constants_class.nsubgrid):
            NMBF_F = extract_subgrid(
                BF_F,
                0,
                constants_class.subgrid_off[i0],
                constants_class,
                use_dask=use_dask,
                nout=1,
            )
            NMBF_BF = prepare_facet(
                NMBF_F, 1, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
            )
            for i1 in range(constants_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][j1] = extract_subgrid(
                    NMBF_BF,
                    1,
                    constants_class.subgrid_off[i1],
                    constants_class,
                    use_dask=use_dask,
                    nout=1,
                )


def facet_to_subgrid_2d_method_3(
    NMBF_NMBF,
    facet,
    constants_class,
    use_dask=False,
):
    """
    Generate subgrid from facet 2D. 3rd Method.

    Approach 3: same as 2, but starts with the vertical direction (axis=1)
                and finishes with the horizontal (axis=0) axis

    :param NMBF_NMBF: TODO ???
    :param facet: 2D numpy array of facets
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: use dask.delayed or not
    """
    for j0, j1 in itertools.product(range(constants_class.nfacet), range(constants_class.nfacet)):
        F_BF = prepare_facet(
            facet[j0][j1], 1, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
        )
        for i1 in range(constants_class.nsubgrid):
            F_NMBF = extract_subgrid(
                F_BF,
                1,
                constants_class.subgrid_off[i1],
                constants_class,
                use_dask=use_dask,
                nout=1,
            )
            BF_NMBF = prepare_facet(
                F_NMBF, 0, constants_class.Fb, constants_class.yP_size, use_dask=use_dask, nout=1
            )
            for i0 in range(constants_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][j1] = extract_subgrid(
                    BF_NMBF,
                    0,
                    constants_class.subgrid_off[i0],
                    constants_class,
                    use_dask=use_dask,
                    nout=1,
                )


def _run_algorithm(
    G_2,
    FG_2,
    distr_fft_class,
    use_dask,
):
    subgrid_2, facet_2 = make_subgrid_and_facet(
        G_2,
        FG_2,
        distr_fft_class,
        dims=2,
        use_dask=use_dask,
    )
    log.info("%s x %s subgrids %s x %s facets", distr_fft_class.nsubgrid, distr_fft_class.nsubgrid, distr_fft_class.nfacet, distr_fft_class.nfacet)

    # ==== Facet to Subgrid ====
    log.info("Executing 2D facet-to-subgrid algorithm")

    # 3 Approaches:
    #   - they differ in when facet is prepared and which axis is run first
    #   - they all give the same result, but with a different speed
    #   - #1 is slowest, because that prepares all facets first, which substantially increases their size
    #     and hence, puts a large amount of data into the following loops

    t = time.time()
    NMBF_NMBF = facet_to_subgrid_2d_method_1(
        facet_2,
        distr_fft_class,
        use_dask=use_dask,
    )
    log.info("%s s", time.time() - t)

    t = time.time()
    facet_to_subgrid_2d_method_2(
        NMBF_NMBF,
        facet_2,
        distr_fft_class,
        use_dask=use_dask,
    )
    log.info("%s s", time.time() - t)

    t = time.time()
    facet_to_subgrid_2d_method_3(
        NMBF_NMBF,
        facet_2,
        distr_fft_class,
        use_dask=use_dask,
    )
    log.info("%s s", time.time() - t)

    # ==== Subgrid to Facet ====
    log.info("Executing 2D subgrid-to-facet algorithm")
    # Celeste: This is based on the original implementation by Peter,
    # and has not involved data redistribution yet.

    t = time.time()
    BMNAF_BMNAF = subgrid_to_facet_algorithm(
        subgrid_2,
        distr_fft_class,
        use_dask=use_dask,
    )
    log.info("%s s", time.time() - t)

    return subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF


def main(to_plot=True, fig_name=None, use_dask=False):
    """
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix into PNG files.
                     If to_plot is set to False, fig_name doesn't have an effect.
    :param use_dask: boolean; use dask?
    """
    log.info("== Chosen configuration")
    distr_fft_class = DistributedFFT(**TARGET_PARS)
    log.info(distr_fft_class)

    if to_plot:
        plot_1(distr_fft_class.pswf, distr_fft_class, fig_name=fig_name)
        plot_2(distr_fft_class, fig_name=fig_name)

    log.info("\n== Generate layout (factes and subgrids")
    # Layout subgrids + facets
    log.info("%d subgrids, %d facets needed to cover" % (distr_fft_class.nsubgrid, distr_fft_class.nfacet))

    log.info("\n== Generate A/B masks and subgrid/facet offsets")
    # Determine subgrid/facet offsets and the appropriate A/B masks for cutting them out.
    # We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

    # adding sources
    add_sources = True
    if add_sources:
        FG_2 = numpy.zeros((distr_fft_class.N, distr_fft_class.N))
        source_count = 1000
        sources = [
            (
                numpy.random.randint(-distr_fft_class.N // 2, distr_fft_class.N // 2 - 1),
                numpy.random.randint(-distr_fft_class.N // 2, distr_fft_class.N // 2 - 1),
                numpy.random.rand() * distr_fft_class.N * distr_fft_class.N / numpy.sqrt(source_count) / 2,
            )
            for _ in range(source_count)
        ]
        for x, y, i in sources:
            FG_2[y + distr_fft_class.N // 2, x + distr_fft_class.N // 2] += i
        G_2 = ifft(FG_2)

    else:
        # without sources
        G_2 = (
            numpy.exp(2j * numpy.pi * numpy.random.rand(distr_fft_class.N, distr_fft_class.N))
            * numpy.random.rand(distr_fft_class.N, distr_fft_class.N)
            / 2
        )
        FG_2 = fft(G_2)

    log.info("Mean grid absolute: %s", numpy.mean(numpy.abs(G_2)))

    if use_dask:
        subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF = _run_algorithm(
            G_2,
            FG_2,
            distr_fft_class,
            use_dask=True,
        )

        subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF = dask.compute(
            subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF
        )

        subgrid_2 = numpy.array(subgrid_2)
        facet_2 = numpy.array(facet_2)
        NMBF_NMBF = numpy.array(NMBF_NMBF)
        BMNAF_BMNAF = numpy.array(BMNAF_BMNAF)

    else:
        subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF = _run_algorithm(
            G_2,
            FG_2,
            distr_fft_class,
            use_dask=False,
        )

    errors_facet_to_subgrid_2d(
        NMBF_NMBF,
        distr_fft_class,
        subgrid_2,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    test_accuracy_facet_to_subgrid(
        distr_fft_class,
        xs=252,
        ys=252,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    errors_subgrid_to_facet_2d(
        BMNAF_BMNAF,
        facet_2,
        distr_fft_class.nfacet,
        distr_fft_class.yB_size,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    test_accuracy_subgrid_to_facet(
        distr_fft_class,
        xs=252,
        ys=252,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    # Testing
    with numpy.printoptions(threshold=numpy.inf):
        print(repr(BMNAF_BMNAF[BMNAF_BMNAF != 0]))
    return subgrid_2, facet_2, NMBF_NMBF, BMNAF_BMNAF


if __name__ == "__main__":
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    client = set_up_dask()
    with performance_report(filename="dask-report-2d.html"):
        main(to_plot=False, use_dask=True)
    tear_down_dask(client)

    # all above needs commenting and this uncommenting if want to run it without dask
    # main(to_plot=False)