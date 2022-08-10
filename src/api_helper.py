# pylint: disable=too-many-arguments
# pylint: disable=consider-using-set-comprehension
"""
Some helper function for Distributed Fourier Transform API
"""


import numpy

from src.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)


def make_subgrid(N, xA_size, sg_off0, sg_A0, sg_off1, sg_A1, sources):
    """make subgrid"""
    return make_subgrid_from_sources(
        sources, N, xA_size, [sg_off0, sg_off1], [sg_A0, sg_A1]
    )


def make_facet(
    N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
):
    """make facet"""
    # Create facet
    return make_facet_from_sources(
        sources, N, yB_size, [facet_off0, facet_off1], [facet_B0, facet_B1]
    )


def check_facet(
    N, facet_off0, facet_B0, facet_off1, facet_B1, approx_facet, sources
):
    """
    check facet using sources
    """
    # Re-generate facet to compare against
    yB_size = approx_facet.shape[0]
    facet = make_facet(
        N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
    )

    # Compare against result
    return numpy.sqrt(numpy.average(numpy.abs(facet - approx_facet) ** 2))


def check_residual(residual_facet):
    """
    check residual image
    """
    return numpy.sqrt(numpy.average(numpy.abs(residual_facet) ** 2))


def check_subgrid(N, sg_off0, sg_A0, sg_off1, sg_A1, approx_subgrid, sources):
    """
    check subgrid using sources
    """
    # Compare against subgrid (normalised)
    subgrid = make_subgrid_from_sources(
        sources, N, approx_subgrid.shape[0], [sg_off0, sg_off1], [sg_A0, sg_A1]
    )
    return numpy.sqrt(numpy.average(numpy.abs(subgrid - approx_subgrid) ** 2))


def sum_and_finish_subgrid(
    distributedFFT,
    NMBF_NMBF_tasks,
    facets_j_list,
    subgrid_mask0,
    subgrid_mask1,
):
    """sum faect contribution and finsh subgrid"""
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distributedFFT.xM_size, distributedFFT.xM_size), dtype=complex
    )

    for (facet_off0, facet_off1), NMBF_NMBF in zip(
        facets_j_list, NMBF_NMBF_tasks
    ):
        summed_facet += distributedFFT.add_facet_contribution(
            distributedFFT.add_facet_contribution(
                NMBF_NMBF, facet_off0, axis=0
            ),
            facet_off1,
            axis=1,
        )
    # Finish
    if (subgrid_mask0 is not None) and (subgrid_mask1 is not None):
        approx_subgrid = distributedFFT.finish_subgrid(
            summed_facet,
            [subgrid_mask0, subgrid_mask1],
        )
    else:
        approx_subgrid = distributedFFT.finish_subgrid(
            summed_facet,
        )
    return approx_subgrid


def prepare_and_split_subgrid(distributedFFT, Fn, facets_j_off, subgrid):
    """prepare NAF_NAF"""
    # Prepare subgrid
    prepared_subgrid = distributedFFT.prepare_subgrid(subgrid)

    # Extract subgrid facet contributions

    NAF_AFs = {
        off0: distributedFFT.extract_subgrid_contrib_to_facet(
            prepared_subgrid, off0, Fn, axis=0
        )
        for off0 in set(off0 for off0, off1 in facets_j_off)
    }
    NAF_NAFs = [
        distributedFFT.extract_subgrid_contrib_to_facet(
            NAF_AFs[off0], off1, Fn, axis=1
        )
        for off0, off1 in facets_j_off
    ]
    return NAF_NAFs


def accumulate_column(
    distributedFFT, NAF_NAF, NAF_MNAF, facet_m0_trunc, subgrid_off1
):
    """update NAF_MNAF"""
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    if NAF_MNAF is None:
        return distributedFFT.add_subgrid_contribution(
            NAF_NAF,
            subgrid_off1,
            facet_m0_trunc,
            axis=1,
        )
    return NAF_MNAF + distributedFFT.add_subgrid_contribution(
        NAF_NAF,
        subgrid_off1,
        facet_m0_trunc,
        axis=1,
    )


def accumulate_facet(
    distributedFFT,
    NAF_MNAF,
    MNAF_BMNAF,
    Fb,
    facet_m0_trunc,
    facet_mask1,
    off0,
):
    """update MNAF_BMNAF"""
    NAF_BMNAF = distributedFFT.finish_facet(NAF_MNAF, facet_mask1, Fb, axis=1)
    if MNAF_BMNAF is None:
        return distributedFFT.add_subgrid_contribution(
            NAF_BMNAF, off0, facet_m0_trunc, axis=0
        )
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    MNAF_BMNAF = MNAF_BMNAF + distributedFFT.add_subgrid_contribution(
        NAF_BMNAF, off0, facet_m0_trunc, axis=0
    )
    return MNAF_BMNAF


def finish_facet(distriFFT, MNAF_BMNAFs, Fb, facet_mask0):
    """wrapper of finish_facet"""
    return distriFFT.finish_facet(
        MNAF_BMNAFs,
        facet_mask0,
        Fb,
        axis=0,
    )


def extract_column(
    distriFFT, BF_F, Fn_task, Fb_task, facet_m0_trunc_task, subgrid_off0
):
    """extract column task"""
    return distriFFT.prepare_facet(
        distriFFT.extract_facet_contrib_to_subgrid(
            BF_F,
            subgrid_off0,
            facet_m0_trunc_task,
            Fn_task,
            axis=0,
        ),
        Fb_task,
        axis=1,
    )
