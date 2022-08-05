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
    distributedFFT, NMBF_NMBF_tasks, base_arrays, i0, i1, facets_light_j_off
):
    """sum faect contribution and finsh subgrid"""
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distributedFFT.xM_size, distributedFFT.xM_size), dtype=complex
    )

    for facets_config_j0 in facets_light_j_off:
        for j0, j1 in facets_config_j0:
            NMBF_NMBF = NMBF_NMBF_tasks[j0][j1]
            summed_facet += distributedFFT.add_facet_contribution(
                distributedFFT.add_facet_contribution(
                    NMBF_NMBF, base_arrays.facet_off[j0], axis=0
                ),
                base_arrays.facet_off[j1],
                axis=1,
            )
    # Finish
    return distributedFFT.finish_subgrid(
        summed_facet,
        [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]],
    )


def prepare_and_split_subgrid(
    distributedFFT, Fn, facets_light_j_off, subgrid, base_arrays_task
):
    """prepare NAF_NAF"""
    # Prepare subgrid
    prepared_subgrid = distributedFFT.prepare_subgrid(subgrid)

    # Extract subgrid facet contributions
    facet_ixs = []
    for facets_config_i0 in facets_light_j_off:
        for facets_config in facets_config_i0:
            facet_ixs.append((facets_config[0], facets_config[1]))
    NAF_AFs = {
        j0: distributedFFT.extract_subgrid_contrib_to_facet(
            prepared_subgrid, base_arrays_task.facet_off[j0], Fn, axis=0
        )
        for j0 in set(j0 for j0, j1 in facet_ixs)
    }
    NAF_NAFs = [
        distributedFFT.extract_subgrid_contrib_to_facet(
            NAF_AFs[j0], base_arrays_task.facet_off[j1], Fn, axis=1
        )
        for j0, j1 in facet_ixs
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
    base_arrays_task,
    j1,
    i0,
):
    """update MNAF_BMNAF"""
    NAF_BMNAF = distributedFFT.finish_facet(
        NAF_MNAF, base_arrays_task.facet_B[j1], Fb, axis=1
    )
    if MNAF_BMNAF is None:
        return distributedFFT.add_subgrid_contribution(
            NAF_BMNAF, distributedFFT.subgrid_off[i0], facet_m0_trunc, axis=0
        )
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    MNAF_BMNAF = MNAF_BMNAF + distributedFFT.add_subgrid_contribution(
        NAF_BMNAF, distributedFFT.subgrid_off[i0], facet_m0_trunc, axis=0
    )
    return MNAF_BMNAF


def finish_facet(distriFFT, MNAF_BMNAFs, base_arrays_task, Fb, j0):
    """wrapper of finish_facet"""
    return distriFFT.finish_facet(
        MNAF_BMNAFs,
        base_arrays_task.facet_B[j0],
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