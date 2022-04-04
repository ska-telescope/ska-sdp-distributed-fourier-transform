#!/usr/bin/env python
# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments
# pylint: disable=redefined-outer-name, too-many-lines
"""
Main algorithm routine.
The functions that conduct the main Dask-implemented algorithm
include the subgrid to facet, and facet to subgrid transformations.
The main function calls all the functions.
"""

import itertools
import logging
import os
import sys
import time

import dask
import dask.array
import numpy

# from distributed import performance_report
from matplotlib import pylab

from src.fourier_transform.algorithm_parameters import SparseFourierTransform
from src.fourier_transform.dask_wrapper import set_up_dask, tear_down_dask
from src.fourier_transform.fourier_algorithm import (
    add_facet_contribution,
    add_subgrid_contribution,
    extract_facet_contrib_to_subgrid,
    extract_subgrid_contrib_to_facet,
    fft,
    finish_facet,
    finish_subgrid,
    ifft,
    make_subgrid_and_facet,
    prepare_facet,
    prepare_subgrid,
)
from src.swift_configs import SWIFT_CONFIGS
from src.utils import (
    errors_facet_to_subgrid_2d,
    errors_subgrid_to_facet_2d,
    plot_pswf,
    plot_work_terms,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Plot setup
pylab.rcParams["figure.figsize"] = 16, 4
pylab.rcParams["image.cmap"] = "viridis"


def _generate_subgrid_contributions(subgrid_2, sparse_ft_class, use_dask):
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
            subgrid_2[i0][i1], use_dask=use_dask, nout=1
        )
        for j0 in range(sparse_ft_class.nfacet):
            NAF_AF = sparse_ft_class.extract_subgrid_contrib_to_facet(
                AF_AF,
                sparse_ft_class.facet_off[j0],
                0,
                use_dask=use_dask,
                nout=1,
            )
            for j1 in range(sparse_ft_class.nfacet):
                subgrid_contrib[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_subgrid_contrib_to_facet(
                    NAF_AF,
                    sparse_ft_class.facet_off[j1],
                    1,
                    use_dask=use_dask,
                    nout=1,
                )
    return subgrid_contrib


def subgrid_to_facet_algorithm(subgrid_2, sparse_ft_class, use_dask=False):
    """
    Generate facets from subgrids.

    :param subgrid_2: 2D numpy array of subgrids
    :param sparse_ft_class: SparseFourierTransform class object
    :param use_dask: use dask.delayed or not

    :return: numpy array of approximate facets
    """
    naf_naf = _generate_subgrid_contributions(
        subgrid_2, sparse_ft_class, use_dask
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
        MNAF_BMNAF = numpy.zeros(
            (sparse_ft_class.yP_size, sparse_ft_class.yB_size), dtype=complex
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NAF_MNAF = numpy.zeros(
                (sparse_ft_class.xM_yN_size, sparse_ft_class.yP_size),
                dtype=complex,
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                if use_dask:
                    NAF_MNAF = NAF_MNAF + dask.array.from_delayed(
                        sparse_ft_class.add_subgrid_contribution(
                            len(NAF_MNAF.shape),
                            naf_naf[i0][i1][j0][j1],
                            sparse_ft_class.subgrid_off[i1],
                            1,
                            use_dask=use_dask,
                            nout=1,
                        ),
                        shape=(
                            sparse_ft_class.xM_yN_size,
                            sparse_ft_class.yP_size,
                        ),
                        dtype=complex,
                    )
                else:
                    NAF_MNAF = (
                        NAF_MNAF
                        + sparse_ft_class.add_subgrid_contribution(
                            len(NAF_MNAF.shape),
                            naf_naf[i0][i1][j0][j1],
                            sparse_ft_class.subgrid_off[i1],
                            1,
                        )
                    )
            NAF_BMNAF = sparse_ft_class.finish_facet(
                NAF_MNAF,
                sparse_ft_class.facet_B[j1],
                1,
                use_dask=use_dask,
                nout=0,
            )
            if use_dask:
                MNAF_BMNAF = MNAF_BMNAF + dask.array.from_delayed(
                    sparse_ft_class.add_subgrid_contribution(
                        len(MNAF_BMNAF.shape),
                        NAF_BMNAF,
                        sparse_ft_class.subgrid_off[i0],
                        0,
                        use_dask=use_dask,
                        nout=1,
                    ),
                    shape=(sparse_ft_class.yP_size, sparse_ft_class.yB_size),
                    dtype=complex,
                )
            else:
                MNAF_BMNAF = (
                    MNAF_BMNAF
                    + sparse_ft_class.add_subgrid_contribution(
                        len(MNAF_BMNAF.shape),
                        NAF_BMNAF,
                        sparse_ft_class.subgrid_off[i0],
                        0,
                        use_dask=use_dask,
                        nout=1,
                    )
                )
        approx_facet[j0][j1] = sparse_ft_class.finish_facet(
            MNAF_BMNAF,
            sparse_ft_class.facet_B[j0],
            0,
            use_dask=use_dask,
            nout=1,
        )

    return approx_facet


@dask.delayed
def single_subgrid_to_facet_contributions(
    subgrid_2_i0_i1, j0, j1, xM_size, facet_off, xM_yN_size, N, Fn
):
    """
    Generate the array of individual subgrid contributions to each facet.

    :param subgrid_2: 2D numpy array of subgrids
    :param j0: coordinate 1 in facets
    :param j1: coordinate 2 in facets
    :param xM_size: added subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param facet_off: Facet offset array
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param N: total image size
    :param Fn: Fourier transform of gridding function

    :return: subgrid contributions
    """
    AF_AF = prepare_subgrid(subgrid_2_i0_i1, xM_size)
    NAF_AF = extract_subgrid_contrib_to_facet(
        AF_AF, facet_off[j0], 0, xM_size, xM_yN_size, N, Fn
    )
    subgrid_contrib = extract_subgrid_contrib_to_facet(
        NAF_AF, facet_off[j1], 1, xM_size, xM_yN_size, N, Fn
    )
    return subgrid_contrib


@dask.delayed
def sum_subgrid_contrib_to_facet(
    naf_naf,
    j0,
    j1,
    nsubgrid,
    xM_yN_size,
    yP_size,
    yB_size,
    subgrid_off,
    xMxN_yP_size,
    xM_yP_size,
    N,
    facet_m0_trunc,
    facet_B,
    Fb,
):
    """
    Combine all sub-grid contributions to a facet.

    :param naf_naf:
    :param j0: coordinate 1 in facets
    :param j1: coordinate 2 in facets
    :param nsubgrid: added subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param yP_size: Facet offset array
    :param yB_size: (padded subgrid size * padding) / N
    :param subgrid_off： Subgrid offset array
    :param xMxN_yP_size：length of the region to be cut out of the
                       prepared facet data (i.e. len(facet_m0_trunc),
                       where facet_m0_trunc is the mask truncated
                       to a facet (image space))
    :param xM_yP_size: (padded subgrid size * padded facet size) / N
    :param N: total image size
    :param facet_m0_trunc: Mask truncated to a facet (image space)
    :param facet_B: Facet mask
    :param Fb: Fourier transform of grid correction function

    :return: facet
    """
    MNAF_BMNAF = numpy.zeros((yP_size, yB_size), dtype=complex)
    for i0 in range(nsubgrid):
        NAF_MNAF = numpy.zeros((xM_yN_size, yP_size), dtype=complex)
        for i1 in range(nsubgrid):
            NAF_MNAF = NAF_MNAF + add_subgrid_contribution(
                len(NAF_MNAF.shape),
                naf_naf[i0][i1],
                subgrid_off[i1],
                1,
                xMxN_yP_size,
                xM_yP_size,
                yP_size,
                N,
                facet_m0_trunc,
            )
        NAF_BMNAF = finish_facet(NAF_MNAF, facet_B[j1], 1, yB_size, Fb)
        MNAF_BMNAF = MNAF_BMNAF + add_subgrid_contribution(
            len(MNAF_BMNAF.shape),
            NAF_BMNAF,
            subgrid_off[i0],
            0,
            xMxN_yP_size,
            xM_yP_size,
            yP_size,
            N,
            facet_m0_trunc,
        )
    approx_facet = finish_facet(MNAF_BMNAF, facet_B[j0], 0, yB_size, Fb)
    return approx_facet


def subgrid_to_facet_algorithm_2(subgrid_list, sparse_ft_class):
    """
    Generate facets from subgrids.

    :param subgrid_list: 2D numpy array of subgrids
    :param sparse_ft_class: SparseFourierTransform class object

    :return: numpy array of approximate facets
    """
    subgrid_contrib = numpy.empty(
        (
            sparse_ft_class.nfacet,
            sparse_ft_class.nfacet,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.nsubgrid,
        ),
    ).tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        for i0, i1 in itertools.product(
            range(sparse_ft_class.nsubgrid), range(sparse_ft_class.nsubgrid)
        ):
            tmp = single_subgrid_to_facet_contributions(
                subgrid_list[i0][i1],
                j0,
                j1,
                sparse_ft_class.xM_size,
                sparse_ft_class.facet_off,
                sparse_ft_class.xM_yN_size,
                sparse_ft_class.N,
                sparse_ft_class.Fn,
            )
            subgrid_contrib[j0][j1][i0][i1] = tmp

    approx_facet = numpy.empty(
        (sparse_ft_class.nfacet, sparse_ft_class.nfacet),
    ).tolist()

    for j0, j1 in itertools.product(
        range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
    ):
        subgrid_contrib_list = subgrid_contrib[j0][j1]
        approx_facet[j0][j1] = sum_subgrid_contrib_to_facet(
            subgrid_contrib_list,
            j0,
            j1,
            sparse_ft_class.nsubgrid,
            sparse_ft_class.xM_yN_size,
            sparse_ft_class.yP_size,
            sparse_ft_class.yB_size,
            sparse_ft_class.subgrid_off,
            sparse_ft_class.xMxN_yP_size,
            sparse_ft_class.xM_yP_size,
            sparse_ft_class.N,
            sparse_ft_class.facet_m0_trunc,
            sparse_ft_class.facet_B,
            sparse_ft_class.Fb,
        )
    return approx_facet


def facet_to_subgrid_2d_method_1(facet, sparse_ft_class, use_dask=False):
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
            facet[j0][j1], 0, use_dask=use_dask, nout=1
        )
        BF_BF = sparse_ft_class.prepare_facet(
            BF_F, 1, use_dask=use_dask, nout=1
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NMBF_BF = sparse_ft_class.extract_facet_contrib_to_subgrid(
                BF_BF,
                0,
                sparse_ft_class.subgrid_off[i0],
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
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_2(facet, sparse_ft_class, use_dask=False):
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
            facet[j0][j1], 0, use_dask=use_dask, nout=1
        )
        for i0 in range(sparse_ft_class.nsubgrid):
            NMBF_F = sparse_ft_class.extract_facet_contrib_to_subgrid(
                BF_F,
                0,
                sparse_ft_class.subgrid_off[i0],
                use_dask=use_dask,
                nout=1,
            )
            NMBF_BF = sparse_ft_class.prepare_facet(
                NMBF_F, 1, use_dask=use_dask, nout=1
            )
            for i1 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    1,
                    sparse_ft_class.subgrid_off[i1],
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, use_dask=use_dask
    )

    return approx_subgrid


def facet_to_subgrid_2d_method_3(facet, sparse_ft_class, use_dask=False):
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
            facet[j0][j1], 1, use_dask=use_dask, nout=1
        )
        for i1 in range(sparse_ft_class.nsubgrid):
            F_NMBF = sparse_ft_class.extract_facet_contrib_to_subgrid(
                F_BF,
                1,
                sparse_ft_class.subgrid_off[i1],
                use_dask=use_dask,
                nout=1,
            )
            BF_NMBF = sparse_ft_class.prepare_facet(
                F_NMBF, 0, use_dask=use_dask, nout=1
            )
            for i0 in range(sparse_ft_class.nsubgrid):
                NMBF_NMBF[i0][i1][j0][
                    j1
                ] = sparse_ft_class.extract_facet_contrib_to_subgrid(
                    BF_NMBF,
                    0,
                    sparse_ft_class.subgrid_off[i0],
                    use_dask=use_dask,
                    nout=1,
                )

    approx_subgrid = generate_approx_subgrid(
        NMBF_NMBF, sparse_ft_class, use_dask=use_dask
    )

    return approx_subgrid


@dask.delayed
def single_facet_to_NMBF_NMBF(
    facet_j_j,
    i0,
    i1,
    Fb,
    subgrid_off,  # i1
    yP_size,
    xMxN_yP_size,
    xM_yP_size,
    xM_yN_size,
    N,
    Fn,
    facet_m0_trunc,
):
    """
    Calculate NMBF_NMBF from a facet

    :param facet_j_j: facet(j,j)
    :param i0: Subgrid coordinate
    :param i1: Subgrid coordinate
    :param Fb: Fourier transform of grid correction function
    :param subgrid_off: Subgrid offset array
    :param yP_size: padded facet size (pad facet with zeros
                    at margins to reach this size)
    :param xMxN_yP_size: length of the region to be cut out of the
                         prepared facet data (i.e. len(facet_m0_trunc),
                         where facet_m0_trunc is the mask truncated
                         to a facet (image space))
    :param xM_yP_size: (padded subgrid size * padded facet size) / N
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param N: total image size
    :param Fn: Fourier transform of gridding function
    :param facet_m0_trunc: Mask truncated to a facet (image space)

    :return: NMBF_NMBF
    """
    # print(facet_j_j.shape)

    F_BF = prepare_facet(facet_j_j, 1, Fb=Fb, yP_size=yP_size)

    F_NMBF = extract_facet_contrib_to_subgrid(
        F_BF,
        1,
        subgrid_off[i1],  # i1
        yP_size,
        xMxN_yP_size,
        xM_yP_size,
        xM_yN_size,
        N,
        Fn,
        facet_m0_trunc,
    )
    BF_NMBF = prepare_facet(F_NMBF, 0, Fb=Fb, yP_size=yP_size)
    NMBF_NMBF = extract_facet_contrib_to_subgrid(
        BF_NMBF,
        0,
        subgrid_off[i0],
        yP_size,
        xMxN_yP_size,
        xM_yP_size,
        xM_yN_size,
        N,
        Fn,
        facet_m0_trunc,
    )
    return NMBF_NMBF


# TODO: need reduce sum
@dask.delayed
def sum_NMBF_NMBF_one_subgrid(
    NMBF_NMBF_facet_list,
    xM_size,
    nfacet,
    facet_off,
    subgrid_A,
    xA_size,
    N,
    i0,
    i1,
):
    """
    Combine all NMBF_NMBF to a subgrid.
    :param NMBF_NMBF_facet_list: NMBF_NMBF list
    :param xM_size: added subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param nfacet: number of facets
    :param facet_off: Facet offset array
    :param subgrid_A: Subgrid mask
    :param xA_size: effective subgrid size
    :param N: total image size
    :param i0: Coordinate 0 in subgrid
    :param i1: Coordinate 1 in subgrid

    :return: Subgrid
    """
    summed_facet = numpy.zeros((xM_size, xM_size), dtype=complex)
    for j0, j1 in itertools.product(range(nfacet), range(nfacet)):
        summed_facet = summed_facet + add_facet_contribution(
            add_facet_contribution(
                NMBF_NMBF_facet_list[j0][j1],
                facet_off[j0],
                axis=0,
                xM_size=xM_size,
                N=N,
            ),
            facet_off[j1],
            axis=1,
            xM_size=xM_size,
            N=N,
        )
    summed_facet = finish_subgrid(
        summed_facet, subgrid_A[i0], subgrid_A[i1], xA_size
    )
    return summed_facet


def facet_to_subgrid_2d_method_3_serial(facet_list, sparse_ft_class):
    """
    Generate subgrid from facet 2D. 3rd Method but serial execution.

    Approach 3: same as 2, but starts with the vertical direction (axis=1)
                and finishes with the horizontal (axis=0) axis

    :param facet: 2D numpy array of facets
    :param sparse_ft_class: SparseFourierTransform class object
    :param use_dask: use dask.delayed or not

    :return: approximate subgrid array (subgrids derived from facets)
    """

    # create NMBF_NMBF task
    print(sparse_ft_class.Fb, "Done")
    NMBF_NMBF_list = []
    for i0 in range(sparse_ft_class.nsubgrid):
        l1 = []
        for i1 in range(sparse_ft_class.nsubgrid):
            l2 = []
            for j0 in range(sparse_ft_class.nfacet):
                l3 = []
                for j1 in range(sparse_ft_class.nfacet):
                    # print(facet_list[j0][j1].shape,i0,i1,j0,j1,"done")
                    tmp = single_facet_to_NMBF_NMBF(
                        facet_list[j0][j1],
                        i0,
                        i1,
                        sparse_ft_class.Fb,
                        sparse_ft_class.subgrid_off,
                        sparse_ft_class.yP_size,
                        sparse_ft_class.xMxN_yP_size,
                        sparse_ft_class.xM_yP_size,
                        sparse_ft_class.xM_yN_size,
                        sparse_ft_class.N,
                        sparse_ft_class.Fn,
                        sparse_ft_class.facet_m0_trunc,
                    )
                    # print(facet_list[j0][j1].shape,i0,i1,j0,j1,"done")
                    l3.append(tmp)
                l2.append(l3)
            l1.append(l2)
        NMBF_NMBF_list.append(l1)

    # exit(0)

    approx_subgrid = []
    for i0 in range(sparse_ft_class.nsubgrid):
        t1 = []
        for i1 in range(sparse_ft_class.nsubgrid):
            NMBF_NMBF_facet_list = NMBF_NMBF_list[i0][i1]
            single_subgrid = sum_NMBF_NMBF_one_subgrid(
                NMBF_NMBF_facet_list,
                sparse_ft_class.xM_size,
                sparse_ft_class.nfacet,
                sparse_ft_class.facet_off,
                sparse_ft_class.subgrid_A,
                sparse_ft_class.xA_size,
                sparse_ft_class.N,
                i0,
                i1,
            )
            t1.append(single_subgrid)
        approx_subgrid.append(t1)
    return approx_subgrid


def generate_approx_subgrid(NMBF_NMBF, sparse_ft_class, use_dask=False):
    """
    Finish generating subgrids from facets.

    :param NMBF_NMBF: array of individual facet contributions
    :param sparse_ft_class: SparseFourierTransform class object
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
            summed_facet = summed_facet.tolist()

        for j0, j1 in itertools.product(
            range(sparse_ft_class.nfacet), range(sparse_ft_class.nfacet)
        ):
            summed_facet = (
                summed_facet
                + sparse_ft_class.add_facet_contribution(
                    sparse_ft_class.add_facet_contribution(
                        NMBF_NMBF[i0][i1][j0][j1],
                        sparse_ft_class.facet_off[j0],
                        axis=0,
                        use_dask=use_dask,
                        nout=1,
                    ),
                    sparse_ft_class.facet_off[j1],
                    axis=1,
                    use_dask=use_dask,
                    nout=1,
                )
            )

        approx_subgrid[i0][i1] = sparse_ft_class.finish_subgrid(
            summed_facet,
            sparse_ft_class.subgrid_A[i0],
            sparse_ft_class.subgrid_A[i1],
            use_dask=use_dask,
            nout=1,
        )

    return approx_subgrid


def _run_algorithm(
    G_2, FG_2, sparse_ft_class, use_dask, version_to_run=3, client=None
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
    if client is None:
        G_2_submit = G_2
        FG_2_submit = FG_2
    else:
        print(sparse_ft_class.Fb, "Done")
        G_2_submit = client.scatter(G_2)
        FG_2_submit = client.scatter(FG_2)
    subgrid_2, facet_2 = make_subgrid_and_facet(
        G_2_submit, FG_2_submit, sparse_ft_class, dims=2, use_dask=use_dask
    )
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
            facet_2, sparse_ft_class, use_dask=use_dask
        )
        log.info("%s s", time.time() - t)

    elif version_to_run == 2:
        # Version #2
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_2(
            facet_2, sparse_ft_class, use_dask=use_dask
        )
        log.info("%s s", time.time() - t)

    elif version_to_run == 3:
        # Version #3
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_3(
            facet_2, sparse_ft_class, use_dask=use_dask
        )
        log.info("%s s", time.time() - t)
    else:
        t = time.time()
        approx_subgrid = facet_to_subgrid_2d_method_3_serial(
            facet_2, sparse_ft_class
        )
        log.info("%s s", time.time() - t)
    # ==== Subgrid to Facet ====
    log.info("Executing 2D subgrid-to-facet algorithm")
    # Celeste: This is based on the original implementation by Peter,
    # and has not involved data redistribution yet.

    t = time.time()
    if version_to_run <= 3:
        approx_facet = subgrid_to_facet_algorithm(
            subgrid_2, sparse_ft_class, use_dask=use_dask
        )
    else:
        approx_facet = subgrid_to_facet_algorithm_2(subgrid_2, sparse_ft_class)
    log.info("%s s", time.time() - t)

    return subgrid_2, facet_2, approx_subgrid, approx_facet


def main(
    fundamental_params,
    to_plot=True,
    fig_name=None,
    use_dask=False,
    client=None,
):
    """

    Main execution function that reads in the configuration,
    generates the source data, and runs the algorithm.

    :param fundamental_params: dict of parameters needed to instantiate
                    src.fourier_transform.algorithm_parameters.DistributedFFT
    :param to_plot: run plotting?
    :param fig_name: If given, figures are saved with this prefix into
                     PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    :param use_dask: boolean; use dask?
    """
    log.info("== Chosen configuration")
    sparse_ft_class = SparseFourierTransform(**fundamental_params)
    log.info(sparse_ft_class)

    if to_plot:
        plot_pswf(sparse_ft_class.pswf, sparse_ft_class, fig_name=fig_name)
        plot_work_terms(sparse_ft_class, fig_name=fig_name)

    log.info("\n== Generate layout (facets and subgrids")
    # Layout subgrids + facets
    log.info(
        "%s subgrids, %s facets needed to cover",
        sparse_ft_class.nsubgrid,
        sparse_ft_class.nfacet,
    )

    log.info("\n== Generate A/B masks and subgrid/facet offsets")
    # Determine subgrid/facet offsets and the appropriate
    # A/B masks for cutting them out.
    # We are aiming for full coverage here:
    #   Every pixel is part of exactly one subgrid / facet.

    # adding sources
    add_sources = True

    # TODO: If we need to process large scale Grids/Facets, the "add_source"
    #       branches need to be modified by using DFT (ORC-1228)
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

    if use_dask:
        subgrid_2, facet_2, approx_subgrid, approx_facet = _run_algorithm(
            G_2, FG_2, sparse_ft_class, use_dask=True, client=client
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
            G_2, FG_2, sparse_ft_class, use_dask=False
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


if __name__ == "__main__":
    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    test_conf = SWIFT_CONFIGS["1k[1]-512-256"]

    client = set_up_dask(scheduler_address=scheduler)
    # with performance_report(filename="dask-report-2d.html"):
    main(test_conf, to_plot=False, use_dask=True, client=client)
    tear_down_dask(client)

    # all above needs commenting and this uncommenting if
    # want to run it without dask
    # main(to_plot=False)
