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
    SparseFourierTransform,
)
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    fft,
    ifft,
    make_subgrid_and_facet,
    make_subgrid_and_facet_from_hdf5,
)
from src.swift_configs import SWIFT_CONFIGS
from src.utils import (
    add_two,
    errors_facet_to_subgrid_2d,
    errors_facet_to_subgrid_2d_dask,
    errors_subgrid_to_facet_2d,
    errors_subgrid_to_facet_2d_dask,
    make_G_2_FG_2,
    make_G_2_FG_2_hdf5,
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
    subgrid_2, sparse_ft_class, base_arrays, use_dask
):
    """
    Generate the array of individual subgrid contributions to each facet.

    :param subgrid_2: 2D numpy array of subgrids
    :param sparse_ft_class: SparseFourierTransform class
    :param use_dask: use dask.delayed or not

    :return: subgrid contributions
    """
    subgrid_contrib = numpy.empty(
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
    if use_dask:
        subgrid_contrib = subgrid_contrib.tolist()
    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        AF_AF = sparse_ft_class.prepare_subgrid(
            subgrid_2[i0][i1],
            use_dask=use_dask,
            nout=1,
        )
        for j0 in range(sparse_ft_class.nfacet):
            NAF_AF = sparse_ft_class.extract_subgrid_contrib_to_facet(
                AF_AF,
                sparse_ft_class.facet_off[j0],
                axis=0,
                base_arrays=base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            for j1 in range(sparse_ft_class.nfacet):
                subgrid_contrib[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_subgrid_contrib_to_facet(
                    NAF_AF,
                    sparse_ft_class.facet_off[j1],
                    axis=1,
                    base_arrays=base_arrays,
                    use_dask=use_dask,
                    nout=1,
                )
    return subgrid_contrib


def subgrid_to_facet_algorithm(
    subgrid_2,
    sparse_ft_class,
    base_arrays,
    use_dask=False,
):
    """
    Generate facets from subgrids.

    :param subgrid_2: 2D numpy array of subgrids
    :param sparse_ft_class: SparseFourierTransform class object
    :param use_dask: use dask.delayed or not

    :return: numpy array of approximate facets
    """
    naf_naf = _generate_subgrid_contributions(
        subgrid_2,
        sparse_ft_class,
        base_arrays,
        use_dask,
    )

    approx_facet = numpy.empty(
        (
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.yB_size,
            sparse_ft_class.yB_size,
        ),
        dtype=complex,
    )
    if use_dask:
        approx_facet = approx_facet.tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        if use_dask:
            MNAF_BMNAF = None
        else:
            MNAF_BMNAF = numpy.zeros(
                (sparse_ft_class.yP_size, sparse_ft_class.yB_size),
                dtype=complex,
            )
        for i0 in range(sparse_ft_class.nsubgrid):
            if use_dask:
                NAF_MNAF = None
            else:
                NAF_MNAF = numpy.zeros(
                    (sparse_ft_class.xM_yN_size, sparse_ft_class.yP_size),
                    dtype=complex,
                )
            for i1 in range(sparse_ft_class.nsubgrid):
                if use_dask:
                    tmp_NAF_MNAF = sparse_ft_class.add_subgrid_contribution(
                        2,
                        naf_naf[i0][i1][j0][j1],
                        sparse_ft_class.subgrid_off[i1],
                        axis=1,
                        base_arrays=base_arrays,
                        use_dask=use_dask,
                        nout=1,
                    )
                    NAF_MNAF = add_two(
                        NAF_MNAF, tmp_NAF_MNAF, use_dask=use_dask, nout=1
                    )
                else:
                    NAF_MNAF = (
                        NAF_MNAF
                        + sparse_ft_class.add_subgrid_contribution(
                            len(NAF_MNAF.shape),
                            naf_naf[i0][i1][j0][j1],
                            sparse_ft_class.subgrid_off[i1],
                            axis=1,
                            base_arrays=base_arrays,
                            use_dask=use_dask,
                            nout=1,
                        )
                    )
            NAF_BMNAF = sparse_ft_class.finish_facet(
                NAF_MNAF,
                j1,
                axis=1,
                base_arrays=base_arrays,
                use_dask=use_dask,
                nout=0,
            )
            if use_dask:
                tmp_MNAF_BMNAF = sparse_ft_class.add_subgrid_contribution(
                    2,
                    NAF_BMNAF,
                    sparse_ft_class.subgrid_off[i0],
                    axis=0,
                    base_arrays=base_arrays,
                    use_dask=use_dask,
                    nout=1,
                )
                MNAF_BMNAF = add_two(
                    MNAF_BMNAF, tmp_MNAF_BMNAF, use_dask=use_dask, nout=1
                )
            else:
                MNAF_BMNAF = (
                    MNAF_BMNAF
                    + sparse_ft_class.add_subgrid_contribution(
                        len(MNAF_BMNAF.shape),
                        NAF_BMNAF,
                        sparse_ft_class.subgrid_off[i0],
                        axis=0,
                        base_arrays=base_arrays,
                        use_dask=use_dask,
                        nout=1,
                    )
                )
        approx_facet[j0][j1] = sparse_ft_class.finish_facet(
            MNAF_BMNAF,
            j0,
            axis=0,
            base_arrays=base_arrays,
            use_dask=use_dask,
            nout=1,
        )

    return approx_facet


def facet_to_subgrid_2d_method_1(
    facet,
    sparse_ft_class,
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
    :param sparse_ft_class: SparseFourierTransform class object
    :param base_arrays: base_arrays object or future is use_dask = True
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """

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
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        BF_F = sparse_ft_class.prepare_facet(
            facet[j0][j1],
            0,
            base_arrays,
            use_dask=use_dask,
            nout=1,
        )
        BF_BF = sparse_ft_class.prepare_facet(
            BF_F,
            1,
            base_arrays,
            use_dask=use_dask,
            nout=1,
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NMBF_BF = sparse_ft_class.extract_facet_contrib_to_subgrid(
                BF_BF,
                0,
                sparse_ft_class.subgrid_off[i0],
                base_arrays=base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    1,
                    sparse_ft_class.subgrid_off[i1],
                    base_arrays=base_arrays,
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, base_arrays, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_2(
    facet,
    sparse_ft_class,
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
    :param sparse_ft_class: SparseFourierTransform class object
    :param base_arrays: base_arrays object or future is use_dask = True
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """
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
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        BF_F = sparse_ft_class.prepare_facet(
            facet[j0][j1],
            0,
            base_arrays,
            use_dask=use_dask,
            nout=1,
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NMBF_F = sparse_ft_class.extract_facet_contrib_to_subgrid(
                BF_F,
                0,
                sparse_ft_class.subgrid_off[i0],
                base_arrays=base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            NMBF_BF = sparse_ft_class.prepare_facet(
                NMBF_F,
                1,
                base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    1,
                    sparse_ft_class.subgrid_off[i1],
                    base_arrays=base_arrays,
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, base_arrays, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_3(
    facet,
    sparse_ft_class,
    base_arrays,
    use_dask=False,
):
    """
    Generate subgrid from facet 2D. 3rd Method.

    Approach 3: same as 2, but starts with the vertical direction (axis=1)
                and finishes with the horizontal (axis=0) axis

    :param facet: 2D numpy array of facets
    :param sparse_ft_class: SparseFourierTransform class object
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """
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
    if use_dask:
        NMBF_NMBF = NMBF_NMBF.tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        F_BF = sparse_ft_class.prepare_facet(
            facet[j0][j1],
            1,
            base_arrays,
            use_dask=use_dask,
            nout=1,
        )
        for i1 in range(sparse_ft_class.nsubgrid):
            F_NMBF = sparse_ft_class.extract_facet_contrib_to_subgrid(
                F_BF,
                1,
                sparse_ft_class.subgrid_off[i1],
                base_arrays=base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            BF_NMBF = sparse_ft_class.prepare_facet(
                F_NMBF,
                0,
                base_arrays,
                use_dask=use_dask,
                nout=1,
            )
            for i0 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    BF_NMBF,
                    0,
                    sparse_ft_class.subgrid_off[i0],
                    base_arrays=base_arrays,
                    use_dask=use_dask,
                    nout=1,
                )
    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, base_arrays, use_dask=use_dask
    )
    return approx_subgrid


def generate_approx_subgrid(
    NMBF_NMBF, sparse_ft_class, base_arrays, use_dask=False
):
    """
    Finish generating subgrids from facets.

    :param NMBF_NMBF: array of individual facet contributions
    :param sparse_ft_class: SparseFourierTransform class object
    :param base_arrays: base_arrays object or future is use_dask = True
    :param use_dask: use dask.delayed or not
    """
    approx_subgrid = numpy.empty(
        (
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.xA_size,
            sparse_ft_class.xA_size,
        ),
        dtype=complex,
    )
    if use_dask:
        approx_subgrid = approx_subgrid.tolist()

    for i0, i1 in itertools.product(
        range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
    ):
        summed_facet = numpy.zeros(
            (sparse_ft_class.xM_size, sparse_ft_class.xM_size), dtype=complex
        )
        if use_dask:
            summed_facet = None

        for j0, j1 in itertools.product(
            range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
        ):
            tmp_axis_0 = sparse_ft_class.add_facet_contribution(
                NMBF_NMBF[i0][i1][j0][j1],
                sparse_ft_class.facet_off[j0],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )
            tmp_facet = sparse_ft_class.add_facet_contribution(
                tmp_axis_0,
                sparse_ft_class.facet_off[j1],
                axis=1,
                use_dask=use_dask,
                nout=1,
            )
            # Add two facets using Dask delayed
            summed_facet = add_two(
                summed_facet, tmp_facet, use_dask=use_dask, nout=1
            )

        approx_subgrid[i0][i1] = sparse_ft_class.finish_subgrid(
            summed_facet,
            subgrid_mask1_idx=i0,
            subgrid_mask2_idx=i1,
            base_arrays=base_arrays,
            use_dask=use_dask,
            nout=1,
        )

    return approx_subgrid


def _run_algorithm(
    subgrid_2,
    facet_2,
    sparse_ft_class,
    base_arrays_submit,
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

    :param G_2: 2D "ground truth" array; to be split into subgrids
    :param FG_2: FFT of G_2; to be split into facets
    :param sparse_ft_class: SparseFourierTransform class object
    :param use_dask: use dask.delayed or not
    :param version_to_run: which facet-to-subgrid version (method)
                           to run: 1, 2, or 3 (if not 1, or 2, it runs 3)
    """

    log.info(
        "%s x %s subgrids %s x %s facets",
        sparse_ft_class.nsubgrid,
        sparse_ft_class.nsubgrid,
        sparse_ft_class.nfacet,
        sparse_ft_class.nfacet,
    )

    # ==== Facet to Subgrid ====
    log.info("Executing 2D facet-to-subgrid algorithm")

    if version_to_run == 1:
        # Version #1
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_1(
            facet_2,
            sparse_ft_class,
            base_arrays_submit,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    elif version_to_run == 2:
        # Version #2
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_2(
            facet_2,
            sparse_ft_class,
            base_arrays_submit,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    else:
        # Version #3
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_3(
            facet_2,
            sparse_ft_class,
            base_arrays_submit,
            use_dask=use_dask,
        )
        log.info("%s s", time.time() - t)

    # ==== Subgrid to Facet ====
    log.info("Executing 2D subgrid-to-facet algorithm")
    # Celeste: This is based on the original implementation by Peter,
    # and has not involved data redistribution yet.

    t = time.time()
    approx_facet = subgrid_to_facet_algorithm(
        subgrid_2, sparse_ft_class, base_arrays_submit, use_dask=use_dask
    )
    log.info("%s s", time.time() - t)

    return subgrid_2, facet_2, approx_subgrid, approx_facet


# pylint: disable=too-many-arguments
def run_distributed_fft(
    base_arrays,
    fundamental_params,
    to_plot=True,
    fig_name=None,
    use_dask=False,
    client=None,
    use_hdf5=False,
    G_2_file=None,
    FG_2_file=None,
    approx_G_2_file=None,
    approx_FG_2_file=None,
    hdf5_chunksize=None,
):
    """

    Main execution function that reads in the configuration,
    generates the source data, and runs the algorithm.

    :param base_arrays: Data arrays
    :param sparse_ft_class: SparseFourierTransform class object
    :param to_plot: run plotting?
    :param fig_name: If given, figures are saved with this prefix into
                     PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    :param use_dask: boolean; use dask?
    """
    sparse_ft_class = SparseFourierTransform(**fundamental_params)
    log.info("== Chosen configuration")
    log.info(sparse_ft_class)

    if to_plot:
        plot_pswf(base_arrays.pswf, sparse_ft_class, fig_name=fig_name)
        plot_work_terms(sparse_ft_class, fig_name=fig_name)

    log.info("\n== Generate layout (facets and subgrids")
    # Layout subgrids + facets
    log.info(
        "%s subgrids, %s facets needed to cover",
        sparse_ft_class.nsubgrid,
        sparse_ft_class.nfacet,
    )

    # make data
    if use_hdf5 and use_dask:
        if use_dask and client is not None:
            base_arrays_submit = client.scatter(base_arrays)

        # there are two path of hdf5 file
        G_2, FG_2 = make_G_2_FG_2_hdf5(
            sparse_ft_class, G_2_file, FG_2_file, hdf5_chunksize, client
        )
        subgrid_2, facet_2 = make_subgrid_and_facet_from_hdf5(
            G_2,
            FG_2,
            sparse_ft_class,
            base_arrays_submit,
            use_dask=use_dask,
        )
    else:
        G_2, FG_2 = make_G_2_FG_2(sparse_ft_class)

        if use_dask and client is not None:
            G_2 = client.scatter(G_2)
            FG_2 = client.scatter(FG_2)
            base_arrays_submit = client.scatter(base_arrays)

        subgrid_2, facet_2 = make_subgrid_and_facet(
            G_2,
            FG_2,
            base_arrays,  # still use object，
            dims=2,
            use_dask=use_dask,
        )

    # run algorithm
    if use_hdf5 and use_dask:
        subgrid_2, facet_2, approx_subgrid, approx_facet = _run_algorithm(
            subgrid_2,
            facet_2,
            sparse_ft_class,
            base_arrays_submit,
            use_dask=True,
        )

        errors_facet_to_subgrid = errors_facet_to_subgrid_2d_dask(
            approx_subgrid,
            sparse_ft_class,
            subgrid_2,
        )

        errors_subgrid_to_facet = errors_subgrid_to_facet_2d_dask(
            approx_facet,
            facet_2,
            sparse_ft_class,
        )

        approx_G_2_file, approx_FG_2_file = write_hdf5(
            approx_subgrid,
            approx_facet,
            approx_G_2_file,
            approx_FG_2_file,
            sparse_ft_class,
            base_arrays_submit,
        )

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

        subgrid_2, facet_2, approx_subgrid, approx_facet = (
            G_2_file,
            FG_2_file,
            approx_G_2_file,
            approx_FG_2_file,
        )

    else:
        if use_dask:
            subgrid_2, facet_2, approx_subgrid, approx_facet = _run_algorithm(
                subgrid_2,
                facet_2,
                sparse_ft_class,
                base_arrays_submit,
                use_dask=True,
            )

            subgrid_2, facet_2, approx_subgrid, approx_facet = dask.compute(
                subgrid_2, facet_2, approx_subgrid, approx_facet, sync=True
            )

            subgrid_2 = numpy.array(subgrid_2)
            facet_2 = numpy.array(facet_2)
            approx_subgrid = numpy.array(approx_subgrid)
            approx_facet = numpy.array(approx_facet)

        else:
            subgrid_2, facet_2, approx_subgrid, approx_facet = _run_algorithm(
                subgrid_2,
                facet_2,
                sparse_ft_class,
                base_arrays,
                use_dask=False,
            )

        errors_facet_to_subgrid_2d(
            approx_subgrid,
            sparse_ft_class,
            subgrid_2,
            to_plot=to_plot,
            fig_name=fig_name,
        )

        errors_subgrid_to_facet_2d(
            approx_facet,
            facet_2,
            sparse_ft_class,
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
        "to the configuration we want to run the algorithm for",
    )

    parser.add_argument(
        "--use_hdf5",
        type=str,
        default="False",
        help="use hdf5 to save G /FG, approx G /FG in large scale",
    )

    parser.add_argument(
        "--hdf5_chunksize", type=int, default=256, help="hdf5 chunksize"
    )

    parser.add_argument(
        "--hdf5_prefix", type=str, default="./", help="hdf5 path prefix"
    )

    parser.add_argument(
        "--g_file", type=str, default="1k_G.hdf5", help="hdf5 G path"
    )

    parser.add_argument(
        "--fg_file", type=str, default="1k_FG.hdf5", help="hdf5 FG path"
    )

    parser.add_argument(
        "--approx_g_file",
        type=str,
        default="1k_approx_G.hdf5",
        help="hdf5 approx_G path",
    )

    parser.add_argument(
        "--approx_fg_file",
        type=str,
        default="1k_approx_FG.hdf5",
        help="hdf5 approx_FG path",
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

    try:
        swift_config = SWIFT_CONFIGS[args.swift_config]
    except KeyError as error:
        raise KeyError(
            "Provided argument does not match any swift configuration keys. "
            "Please consult src/swift_configs.py for available options."
        ) from error

    base_arrays_class = BaseArrays(**swift_config)
    # We need to call scipy.special.pro_ang1 function before setting up Dask
    # context. Detailed information could be found at Jira ORC-1214
    _ = base_arrays_class.pswf

    client_dask = set_up_dask(scheduler_address=scheduler)
    with performance_report(filename="dask-report-2d.html"):
        run_distributed_fft(
            base_arrays_class,
            swift_config,
            to_plot=False,
            use_dask=True,
            client=client_dask,
            use_hdf5=args.use_hdf5 == "True",
            G_2_file=args.hdf5_prefix + args.g_file,
            FG_2_file=args.hdf5_prefix + args.fg_file,
            approx_G_2_file=args.hdf5_prefix + args.approx_g_file,
            approx_FG_2_file=args.hdf5_prefix + args.approx_fg_file,
            hdf5_chunksize=args.hdf5_chunksize,
        )
    tear_down_dask(client_dask)

    # all above needs commenting and this uncommenting if
    # want to run it without dask
    # main(to_plot=False)


if __name__ == "__main__":
    dfft_parser = cli_parser()
    parsed_args = dfft_parser.parse_args()
    main(parsed_args)
