#!/usr/bin/env python
# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments
"""
Main algorithm routine.
The functions that conduct the main Dask-implemented algorithm
include the subgrid to facet, and facet to subgrid transformations.
The main function calls all the functions.
"""

import argparse
import itertools
import logging
import os
import sys
import time

import dask
import dask.array
import numpy
from distributed import performance_report
from matplotlib import pylab

from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_and_facet,
    make_subgrid_and_facet_from_hdf5,
    make_subgrid_from_sources,
)
from src.swift_configs import SWIFT_CONFIGS
from src.utils import (
    add_two,
    errors_facet_to_subgrid_2d,
    errors_facet_to_subgrid_2d_dask,
    errors_subgrid_to_facet_2d,
    errors_subgrid_to_facet_2d_dask,
    generate_input_data,
    plot_pswf,
    plot_work_terms,
    write_hdf5,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Plot setup
pylab.rcParams["figure.figsize"] = 16, 4
pylab.rcParams["image.cmap"] = "viridis"


def _generate_subgrid_contributions(
    subgrid_2, distr_fft_class, base_arrays, use_dask
):
    """
    Generate the array of individual subgrid contributions to each facet.

    :param subgrid_2: 2D numpy array of subgrids
    :param distr_fft_class: StreamingDistributedFFT class
    :param base_arrays: BaseArrays class
    :param use_dask: use dask.delayed or not

    :return: subgrid contributions
    """
    subgrid_contrib = numpy.empty(
        (
            distr_fft_class.nsubgrid,
            distr_fft_class.nsubgrid,
            distr_fft_class.nfacet,
            distr_fft_class.nfacet,
            distr_fft_class.xM_yN_size,
            distr_fft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    if use_dask:
        subgrid_contrib = subgrid_contrib.tolist()
    for i0, i1 in itertools.product(
        range(distr_fft_class.nsubgrid), range(distr_fft_class.nsubgrid)
    ):
        AF_AF = distr_fft_class.prepare_subgrid(
            subgrid_2[i0][i1],
            use_dask=use_dask,
            nout=1,
        )
        for j0 in range(distr_fft_class.nfacet):
            NAF_AF = distr_fft_class.extract_subgrid_contrib_to_facet(
                AF_AF,
                distr_fft_class.facet_off[j0],
                base_arrays.Fn,
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            for j1 in range(distr_fft_class.nfacet):
                subgrid_contrib[i0][i1][j0][
                    j1
                ] = distr_fft_class.extract_subgrid_contrib_to_facet(
                    NAF_AF,
                    distr_fft_class.facet_off[j1],
                    base_arrays.Fn,
                    axis=1,
                    use_dask=use_dask,
                    nout=1,
                )
    return subgrid_contrib


def subgrid_to_facet_algorithm(
    subgrid_2,
    distr_fft_class,
    base_arrays,
    use_dask=False,
):
    """
    Generate facets from subgrids.

    :param subgrid_2: 2D numpy array of subgrids
    :param distr_fft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class
    :param use_dask: use dask.delayed or not

    :return: numpy array of approximate facets
    """
    naf_naf = _generate_subgrid_contributions(
        subgrid_2,
        distr_fft_class,
        base_arrays,
        use_dask,
    )

    approx_facet = numpy.empty(
        (
            distr_fft_class.nfacet,
            distr_fft_class.nfacet,
            distr_fft_class.yB_size,
            distr_fft_class.yB_size,
        ),
        dtype=complex,
    )
    if use_dask:
        approx_facet = approx_facet.tolist()

    for j0, j1 in itertools.product(
        range(distr_fft_class.nfacet), range(distr_fft_class.nfacet)
    ):
        if use_dask:
            MNAF_BMNAF = None
        else:
            MNAF_BMNAF = numpy.zeros(
                (distr_fft_class.yP_size, distr_fft_class.yB_size),
                dtype=complex,
            )
        for i0 in range(distr_fft_class.nsubgrid):
            if use_dask:
                NAF_MNAF = None
            else:
                NAF_MNAF = numpy.zeros(
                    (distr_fft_class.xM_yN_size, distr_fft_class.yP_size),
                    dtype=complex,
                )
            for i1 in range(distr_fft_class.nsubgrid):
                if use_dask:
                    tmp_NAF_MNAF = distr_fft_class.add_subgrid_contribution(
                        naf_naf[i0][i1][j0][j1],
                        distr_fft_class.subgrid_off[i1],
                        base_arrays.facet_m0_trunc,
                        axis=1,
                        use_dask=use_dask,
                        nout=1,
                    )
                    NAF_MNAF = add_two(
                        NAF_MNAF, tmp_NAF_MNAF, use_dask=use_dask, nout=1
                    )
                else:
                    NAF_MNAF = (
                        NAF_MNAF
                        + distr_fft_class.add_subgrid_contribution(
                            naf_naf[i0][i1][j0][j1],
                            distr_fft_class.subgrid_off[i1],
                            base_arrays.facet_m0_trunc,
                            axis=1,
                            use_dask=use_dask,
                            nout=1,
                        )
                    )
            NAF_BMNAF = distr_fft_class.finish_facet(
                NAF_MNAF,
                base_arrays.facet_B[j1],
                base_arrays.Fb,
                axis=1,
                use_dask=use_dask,
                nout=0,
            )
            if use_dask:
                tmp_MNAF_BMNAF = distr_fft_class.add_subgrid_contribution(
                    NAF_BMNAF,
                    distr_fft_class.subgrid_off[i0],
                    base_arrays.facet_m0_trunc,
                    axis=0,
                    use_dask=use_dask,
                    nout=1,
                )
                MNAF_BMNAF = add_two(
                    MNAF_BMNAF, tmp_MNAF_BMNAF, use_dask=use_dask, nout=1
                )
            else:
                MNAF_BMNAF = (
                    MNAF_BMNAF
                    + distr_fft_class.add_subgrid_contribution(
                        NAF_BMNAF,
                        distr_fft_class.subgrid_off[i0],
                        base_arrays.facet_m0_trunc,
                        axis=0,
                        use_dask=use_dask,
                        nout=1,
                    )
                )
        approx_facet[j0][j1] = distr_fft_class.finish_facet(
            MNAF_BMNAF,
            base_arrays.facet_B[j0],
            base_arrays.Fb,
            axis=0,
            use_dask=use_dask,
            nout=1,
        )

    return approx_facet


def facet_to_subgrid_2d_method_1(
    facet,
    distr_ft_class,
    base_arrays,
    use_dask=False,
):
    """
    Generate subgrid from facet 2D. 1st Method.

    Approach 1: do prepare_facet step across both axes first,
                then go into the loop over subgrids horizontally
                (axis=0) and within that, loop over subgrids
                vertically (axis=1) and do the extract_subgrid
                step in these two directions

    Having those operations separately means that we can shuffle
    things around quite a bit without affecting the result.
    The obvious first choice might be to do all facet-preparation
    up-front, as this allows us to share the computation across all subgrids

    :param facet: 2D numpy array of facets
    :param distr_ft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """

    NMBF_NMBF = numpy.empty(
        (
            distr_ft_class.nsubgrid,
            distr_ft_class.nsubgrid,
            distr_ft_class.nfacet,
            distr_ft_class.nfacet,
            distr_ft_class.xM_yN_size,
            distr_ft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(distr_ft_class.nfacet), range(distr_ft_class.nfacet)
    ):
        BF_F = distr_ft_class.prepare_facet(
            facet[j0][j1],
            base_arrays.Fb,
            axis=0,
            use_dask=use_dask,
            nout=1,
        )
        BF_BF = distr_ft_class.prepare_facet(
            BF_F,
            base_arrays.Fb,
            axis=1,
            use_dask=use_dask,
            nout=1,
        )
        for i0 in range(distr_ft_class.nsubgrid):
            NMBF_BF = distr_ft_class.extract_facet_contrib_to_subgrid(
                BF_BF,
                distr_ft_class.subgrid_off[i0],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            for i1 in range(distr_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = distr_ft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    distr_ft_class.subgrid_off[i1],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=1,
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, distr_ft_class, base_arrays, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_2(
    facet,
    distr_fft_class,
    base_arrays,
    use_dask=False,
):
    """
    Approach 2: First, do prepare_facet on the horizontal axis
                (axis=0), then for loop for the horizontal subgrid direction,
                and do extract_subgrid within same loop do prepare_facet
                in the vertical case (axis=1), then go into the fila subgrid
                loop in the vertical dir and do extract_subgrid for that

    However, remember that `prepare_facet` increases the amount of data
    involved, which in turn means that we need to shuffle more data through
    subsequent computations. Therefore it is actually more efficient to first
    do the subgrid-specific reduction, and *then* continue with the (constant)
    facet preparation along the other axis. We can tackle both axes in whatever
    order we like, it doesn't make a difference for the result.

    :param facet: 2D numpy array of facets
    :param distr_fft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """
    NMBF_NMBF = numpy.empty(
        (
            distr_fft_class.nsubgrid,
            distr_fft_class.nsubgrid,
            distr_fft_class.nfacet,
            distr_fft_class.nfacet,
            distr_fft_class.xM_yN_size,
            distr_fft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(distr_fft_class.nfacet), range(distr_fft_class.nfacet)
    ):
        BF_F = distr_fft_class.prepare_facet(
            facet[j0][j1],
            base_arrays.Fb,
            axis=0,
            use_dask=use_dask,
            nout=1,
        )
        for i0 in range(distr_fft_class.nsubgrid):
            NMBF_F = distr_fft_class.extract_facet_contrib_to_subgrid(
                BF_F,
                distr_fft_class.subgrid_off[i0],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            NMBF_BF = distr_fft_class.prepare_facet(
                NMBF_F,
                base_arrays.Fb,
                axis=1,
                use_dask=use_dask,
                nout=1,
            )
            for i1 in range(distr_fft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = distr_fft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    distr_fft_class.subgrid_off[i1],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=1,
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, distr_fft_class, base_arrays, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_3(
    facet,
    distr_fft_class,
    base_arrays,
    use_dask=False,
):
    """
    Generate subgrid from facet 2D. 3rd Method.

    Approach 3: same as 2, but starts with the vertical direction (axis=1)
                and finishes with the horizontal (axis=0) axis

    :param facet: 2D numpy array of facets
    :param distr_fft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """
    NMBF_NMBF = numpy.empty(
        (
            distr_fft_class.nsubgrid,
            distr_fft_class.nsubgrid,
            distr_fft_class.nfacet,
            distr_fft_class.nfacet,
            distr_fft_class.xM_yN_size,
            distr_fft_class.xM_yN_size,
        ),
        dtype=complex,
    )
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(distr_fft_class.nfacet), range(distr_fft_class.nfacet)
    ):
        F_BF = distr_fft_class.prepare_facet(
            facet[j0][j1],
            base_arrays.Fb,
            axis=1,
            use_dask=use_dask,
            nout=1,
        )
        for i1 in range(distr_fft_class.nsubgrid):
            F_NMBF = distr_fft_class.extract_facet_contrib_to_subgrid(
                F_BF,
                distr_fft_class.subgrid_off[i1],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                axis=1,
                use_dask=use_dask,
                nout=1,
            )
            BF_NMBF = distr_fft_class.prepare_facet(
                F_NMBF,
                base_arrays.Fb,
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            for i0 in range(distr_fft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = distr_fft_class.extract_facet_contrib_to_subgrid(
                    BF_NMBF,
                    distr_fft_class.subgrid_off[i0],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=0,
                    use_dask=use_dask,
                    nout=1,
                )
    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, distr_fft_class, base_arrays, use_dask=use_dask
    )
    return approx_subgrid


def generate_approx_subgrid(
    NMBF_NMBF, distr_fft_class, base_arrays, use_dask=False
):
    """
    Finish generating subgrids from facets.

    :param NMBF_NMBF: array of individual facet contributions
    :param distr_fft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param use_dask: use dask.delayed or not
    """
    approx_subgrid = numpy.empty(
        (
            distr_fft_class.nsubgrid,
            distr_fft_class.nsubgrid,
            distr_fft_class.xA_size,
            distr_fft_class.xA_size,
        ),
        dtype=complex,
    )
    if use_dask:
        approx_subgrid = approx_subgrid.tolist()

    for i0, i1 in itertools.product(
        range(distr_fft_class.nsubgrid), range(distr_fft_class.nsubgrid)
    ):
        summed_facet = numpy.zeros(
            (distr_fft_class.xM_size, distr_fft_class.xM_size), dtype=complex
        )
        if use_dask:
            summed_facet = None

        for j0, j1 in itertools.product(
            range(distr_fft_class.nfacet), range(distr_fft_class.nfacet)
        ):
            tmp_axis_0 = distr_fft_class.add_facet_contribution(
                NMBF_NMBF[i0][i1][j0][j1],
                distr_fft_class.facet_off[j0],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            tmp_facet = distr_fft_class.add_facet_contribution(
                tmp_axis_0,
                distr_fft_class.facet_off[j1],
                axis=1,
                use_dask=use_dask,
                nout=1,
            )
            # Add two facets using Dask delayed (if use_dask = True)
            summed_facet = add_two(
                summed_facet, tmp_facet, use_dask=use_dask, nout=1
            )

        approx_subgrid[i0][i1] = distr_fft_class.finish_subgrid(
            summed_facet,
            [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]],
            use_dask=use_dask,
            nout=1,
        )

    return approx_subgrid


def _run_algorithm(
    subgrid_2,
    facet_2,
    distr_fft_class,
    base_arrays,
    use_dask,
    version_to_run=3,
):
    """
    Run facet-to-subgrid and subgrid-to-facet algorithm.

    Facet-to-subgrid has three versions, which iterate through facets
    and subgrids in different ways. They differ in how long they run for.
    The three approaches:
      - differ in when facet is prepared and which axis is run first
      - all give the same result, but with a different speed
      - #1 is slowest, because that prepares all facets first,
        which substantially increases their size and hence, puts a
        large amount of data into the following loops

    Subgrid-to-facet only has a single version.

    :param subgrid_2: 2D numpy array of subgrids
    :param facet_2: 2D numpy array of facets
    :param distr_fft_class: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param use_dask: use dask.delayed or not
    :param version_to_run: which facet-to-subgrid version (method)
                           to run: 1, 2, or 3 (if not 1, or 2, it runs 3)
    """
    log.info(
        "%s x %s subgrids %s x %s facets",
        distr_fft_class.nsubgrid,
        distr_fft_class.nsubgrid,
        distr_fft_class.nfacet,
        distr_fft_class.nfacet,
    )

    # ==== Facet to Subgrid ====
    log.info("Executing 2D facet-to-subgrid algorithm")

    if version_to_run == 1:
        # Version #1
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_1(
            facet_2,
            distr_fft_class,
            base_arrays,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    elif version_to_run == 2:
        # Version #2
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_2(
            facet_2,
            distr_fft_class,
            base_arrays,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    else:
        # Version #3
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_3(
            facet_2,
            distr_fft_class,
            base_arrays,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    # ==== Subgrid to Facet ====
    log.info("Executing 2D subgrid-to-facet algorithm")
    # Celeste: This is based on the original implementation by Peter,
    # and has not involved data redistribution yet.

    t = time.time()
    approx_facet = subgrid_to_facet_algorithm(
        subgrid_2, distr_fft_class, base_arrays, use_dask=use_dask
    )
    log.info("%s s", time.time() - t)

    return approx_subgrid, approx_facet


# pylint: disable=too-many-arguments
def run_distributed_fft(
    fundamental_params,
    to_plot=True,
    fig_name=None,
    use_dask=False,
    client=None,
    use_hdf5=False,
    hdf5_prefix=None,
    hdf5_chunksize_G=None,
    hdf5_chunksize_FG=None,
    generate_generic=True,
    source_number=10,
    facet_to_subgrid_method=3,
):
    """
    Main execution function that reads in the configuration,
    generates the source data, and runs the algorithm.

    :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py
    :param to_plot: run plotting?
    :param fig_name: If given, figures are saved with this prefix into
                     PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    :param use_dask: boolean; use Dask?
    :param client: Dask client or None
    :param use_hdf5: use Hdf5?
    :param hdf5_prefix: hdf5 path prefix
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data
    :param generate_generic: Whether to generate generic input data
                            (with random sources)
    :param source_number: Number of random sources to add to input data
    :param facet_to_subgrid_method: which method to run
                                    the facet to subgrid algorithm

    :return: subgrid_2, facet_2, approx_subgrid, approx_facet
                when use_hdf5=False
             subgrid_2_file, facet_2_file, approx_subgrid_2_file,
                approx_facet_2_file when use_hdf5=True
    """
    base_arrays = BaseArrays(**fundamental_params)
    distr_fft = StreamingDistributedFFT(**fundamental_params)

    log.info("== Chosen configuration")
    log.info(distr_fft)

    if to_plot:
        plot_pswf(distr_fft, fig_name=fig_name)
        plot_work_terms(distr_fft, fig_name=fig_name)

    log.info("\n== Generated layout (facets and subgrids): \n")
    log.info(
        "%s subgrids, %s facets needed to cover the grid and image space",
        distr_fft.nsubgrid,
        distr_fft.nfacet,
    )

    # The branch of using HDF5
    if use_hdf5:
        G_2_file = f"{hdf5_prefix}/G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
        FG_2_file = f"{hdf5_prefix}/FG_{base_arrays.N}_{hdf5_chunksize_FG}.h5"
        approx_G_2_file = (
            f"{hdf5_prefix}/approx_G_{base_arrays.N}_{hdf5_chunksize_G}.h5"
        )
        approx_FG_2_file = (
            f"{hdf5_prefix}/approx_FG_{base_arrays.N}_{hdf5_chunksize_FG}.h5"
        )

        for hdf5_file in [G_2_file, FG_2_file]:
            if os.path.exists(hdf5_file):
                log.info("Using hdf5 file: %s", hdf5_file)
            else:
                raise FileNotFoundError(
                    f"Please check if the hdf5 data is in the {hdf5_file}"
                )

        subgrid_2, facet_2 = make_subgrid_and_facet_from_hdf5(
            G_2_file,
            FG_2_file,
            base_arrays,
            use_dask=use_dask,
        )
        approx_subgrid, approx_facet = _run_algorithm(
            subgrid_2,
            facet_2,
            distr_fft,
            base_arrays,
            use_dask=use_dask,
            version_to_run=facet_to_subgrid_method,
        )

        errors_facet_to_subgrid = errors_facet_to_subgrid_2d_dask(
            approx_subgrid,
            distr_fft,
            subgrid_2,
            use_dask=use_dask,
        )

        errors_subgrid_to_facet = errors_subgrid_to_facet_2d_dask(
            approx_facet,
            facet_2,
            distr_fft,
            use_dask=use_dask,
        )

        approx_G_2_file, approx_FG_2_file = write_hdf5(
            approx_subgrid,
            approx_facet,
            approx_G_2_file,
            approx_FG_2_file,
            base_arrays,
            hdf5_chunksize_G,
            hdf5_chunksize_FG,
            use_dask=use_dask,
        )

        if use_dask:
            (
                errors_facet_to_subgrid,
                errors_subgrid_to_facet,
                approx_G_2_file,
                approx_FG_2_file,
            ) = dask.compute(
                errors_facet_to_subgrid,
                errors_subgrid_to_facet,
                approx_G_2_file,
                approx_FG_2_file,
                sync=True,
            )

        log.info(
            "errors_facet_to_subgrid RMSE: %s (image: %s)",
            errors_facet_to_subgrid[0],
            errors_facet_to_subgrid[1],
        )

        log.info(
            "errors_subgrid_to_facet RMSE: %s (image: %s)",
            errors_subgrid_to_facet[0],
            errors_subgrid_to_facet[1],
        )

        return G_2_file, FG_2_file, approx_G_2_file, approx_FG_2_file

    G_2, FG_2 = generate_input_data(distr_fft, source_count=source_number)

    if use_dask and client is not None:
        G_2 = client.scatter(G_2)
        FG_2 = client.scatter(FG_2)

    if generate_generic:
        subgrid_2, facet_2 = make_subgrid_and_facet(
            G_2,
            FG_2,
            base_arrays,  # still use object，
            dims=2,
            use_dask=use_dask,
        )
    else:
        # Make facets and subgrids containing just one source
        sources = [(1, 1, 0)]
        if use_dask:
            facet_2 = [
                [
                    dask.delayed(make_facet_from_sources)(
                        sources,
                        base_arrays.N,
                        base_arrays.yB_size,
                        [distr_fft.facet_off[j0], distr_fft.facet_off[j1]],
                        [base_arrays.facet_B[j0], base_arrays.facet_B[j1]],
                    )
                    for j1 in range(distr_fft.nfacet)
                ]
                for j0 in range(distr_fft.nfacet)
            ]

            subgrid_2 = [
                [
                    dask.delayed(make_subgrid_from_sources)(
                        sources,
                        base_arrays.N,
                        base_arrays.xA_size,
                        [distr_fft.subgrid_off[j0], distr_fft.subgrid_off[j1]],
                        [base_arrays.subgrid_A[j0], base_arrays.subgrid_A[j1]],
                    )
                    for j1 in range(distr_fft.nsubgrid)
                ]
                for j0 in range(distr_fft.nsubgrid)
            ]
        else:
            facet_2 = [
                [
                    make_facet_from_sources(
                        sources,
                        base_arrays.N,
                        base_arrays.yB_size,
                        [distr_fft.facet_off[j0], distr_fft.facet_off[j1]],
                        [base_arrays.facet_B[j0], base_arrays.facet_B[j1]],
                    )
                    for j1 in range(distr_fft.nfacet)
                ]
                for j0 in range(distr_fft.nfacet)
            ]

            subgrid_2 = [
                [
                    make_subgrid_from_sources(
                        sources,
                        base_arrays.N,
                        base_arrays.xA_size,
                        [distr_fft.subgrid_off[j0], distr_fft.subgrid_off[j1]],
                        [base_arrays.subgrid_A[j0], base_arrays.subgrid_A[j1]],
                    )
                    for j1 in range(distr_fft.nsubgrid)
                ]
                for j0 in range(distr_fft.nsubgrid)
            ]

    if use_dask:
        approx_subgrid, approx_facet = _run_algorithm(
            subgrid_2,
            facet_2,
            distr_fft,
            base_arrays,
            use_dask=True,
            version_to_run=facet_to_subgrid_method,
        )

        subgrid_2, facet_2, approx_subgrid, approx_facet = dask.compute(
            subgrid_2, facet_2, approx_subgrid, approx_facet, sync=True
        )

        subgrid_2 = numpy.array(subgrid_2)
        facet_2 = numpy.array(facet_2)
        approx_subgrid = numpy.array(approx_subgrid)
        approx_facet = numpy.array(approx_facet)

    else:
        approx_subgrid, approx_facet = _run_algorithm(
            subgrid_2,
            facet_2,
            distr_fft,
            base_arrays,
            use_dask=False,
            version_to_run=facet_to_subgrid_method,
        )

    errors_facet_to_subgrid_2d(
        approx_subgrid,
        distr_fft,
        subgrid_2,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    errors_subgrid_to_facet_2d(
        approx_facet,
        facet_2,
        distr_fft,
        to_plot=to_plot,
        fig_name=fig_name,
    )

    return subgrid_2, facet_2, approx_subgrid, approx_facet


def cli_parser():
    """
    Parse command line arguments

    :return: argparse
    """
    parser = argparse.ArgumentParser(
        description="Distributed Fast Fourier Transform",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--swift_config",
        type=str,
        default="1k[1]-512-256",
        help="Dictionary key from swift_configs.py corresponding "
        "to the configuration we want to run the algorithm for."
        "If coma-separated list of strings, then the code "
        "will iterate through each key. "
        "e.g. '12k[1]-n6k-512,10k[1]-n5k-512,8k[1]-n4k-512'",
    )

    parser.add_argument(
        "--use_hdf5",
        type=str,
        default="False",
        help="use hdf5 to save G /FG, approx G /FG in large scale",
    )

    parser.add_argument(
        "--hdf5_chunksize_G",
        type=int,
        default=256,
        help="hdf5 chunksize for G",
    )

    parser.add_argument(
        "--hdf5_chunksize_FG",
        type=int,
        default=256,
        help="hdf5 chunksize for FG",
    )

    parser.add_argument(
        "--hdf5_prefix", type=str, default="./", help="hdf5 path prefix"
    )

    parser.add_argument(
        "--generate_generic_input",
        type=str,
        default="True",
        help="Whether to generate generic input data (with random sources)",
    )

    parser.add_argument(
        "--source_number",
        type=int,
        default=10,
        help="Number of random sources to add to input data",
    )

    parser.add_argument(
        "--facet_to_subgrid_method",
        type=str,
        default="3",
        help="which facet to subgrid method to run. "
        "Options are 1,2 and 3, see documentation for details",
    )

    return parser


def main(args):
    """
    Main function to run the Distributed FFT
    """
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    swift_config_keys = args.swift_config.split(",")
    # check that all the keys are valid
    for c in swift_config_keys:
        try:
            SWIFT_CONFIGS[c]
        except KeyError as error:
            raise KeyError(
                f"Provided argument ({c}) does not match any swift "
                f"configuration keys. Please consult src/swift_configs.py "
                f"for available options."
            ) from error

    try:
        version = int(args.facet_to_subgrid_method)
    except ValueError:
        log.info("Invalid facet to subgrid method. Use default instead.")
        version = 3

    dask_client = set_up_dask(scheduler_address=scheduler)

    for config_key in swift_config_keys:
        log.info("Running for swift-config: %s", config_key)

        with performance_report(filename=f"dask-report-{config_key}.html"):
            run_distributed_fft(
                SWIFT_CONFIGS[config_key],
                to_plot=False,
                use_dask=True,
                client=dask_client,
                use_hdf5=args.use_hdf5 == "True",
                hdf5_prefix=args.hdf5_prefix,
                hdf5_chunksize_G=args.hdf5_chunksize_G,
                hdf5_chunksize_FG=args.hdf5_chunksize_FG,
                generate_generic=args.generate_generic_data,
                source_number=args.source_number,
                facet_to_subgrid_method=version,
            )
            dask_client.restart()
    tear_down_dask(dask_client)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
