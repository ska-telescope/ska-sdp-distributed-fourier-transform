# pylint: disable=too-many-arguments
# pylint: disable=consider-using-set-comprehension
"""
Some helper function for Distributed Fourier Transform API
"""

import numpy

from .fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)


def make_subgrid(N, xA_size, sg_off0, sg_A0, sg_off1, sg_A1, sources):
    """make subgrid"""
    if isinstance(sg_A0, list) and isinstance(sg_A1, list):
        sg_A0 = make_mask_from_slice(sg_A0[0], sg_A0[1])
        sg_A1 = make_mask_from_slice(sg_A1[0], sg_A1[1])
    return make_subgrid_from_sources(
        sources, N, xA_size, [sg_off0, sg_off1], [sg_A0, sg_A1]
    )


def make_facet(
    N, yB_size, facet_off0, facet_B0, facet_off1, facet_B1, sources
):
    """make facet"""
    # Create facet
    if isinstance(facet_B0, list) and isinstance(facet_B1, list):
        facet_B0 = make_mask_from_slice(facet_B0[0], facet_B0[1])
        facet_B1 = make_mask_from_slice(facet_B1[0], facet_B1[1])

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

    if isinstance(facet_B0, list) and isinstance(facet_B1, list):
        facet_B0 = make_mask_from_slice(facet_B0[0], facet_B0[1])
        facet_B1 = make_mask_from_slice(facet_B1[0], facet_B1[1])

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
    if isinstance(sg_A0, list) and isinstance(sg_A1, list):
        sg_A0 = make_mask_from_slice(sg_A0[0], sg_A0[1])
        sg_A1 = make_mask_from_slice(sg_A1[0], sg_A1[1])
    subgrid = make_subgrid_from_sources(
        sources, N, approx_subgrid.shape[0], [sg_off0, sg_off1], [sg_A0, sg_A1]
    )
    return numpy.sqrt(numpy.average(numpy.abs(subgrid - approx_subgrid) ** 2))


def sum_and_finish_subgrid(
    distributedFFT,
    NMBF_NMBF_tasks,
    Fn,
    facets_config_list,
    subgrid_config,
):
    """sum facet contribution and finsh subgrid"""
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distributedFFT.xM_size, distributedFFT.xM_size), dtype=complex
    )

    for facets_config, NMBF_NMBF in zip(facets_config_list, NMBF_NMBF_tasks):
        summed_facet += distributedFFT.add_facet_contribution(
            distributedFFT.add_facet_contribution(
                NMBF_NMBF, facets_config.off0, Fn, axis=0
            ),
            facets_config.off1,
            Fn,
            axis=1,
        )
    # Finish
    subgrid_mask0 = subgrid_config.mask0
    subgrid_mask1 = subgrid_config.mask1
    if (subgrid_mask0 is not None) and (subgrid_mask1 is not None):
        if isinstance(subgrid_mask0, list) and isinstance(subgrid_mask1, list):
            subgrid_mask0 = make_mask_from_slice(
                subgrid_mask0[0], subgrid_mask0[1]
            )
            subgrid_mask1 = make_mask_from_slice(
                subgrid_mask1[0], subgrid_mask1[1]
            )
        approx_subgrid = distributedFFT.finish_subgrid(
            summed_facet,
            [subgrid_config.off0, subgrid_config.off1],
            [subgrid_mask0, subgrid_mask1],
        )
    else:
        approx_subgrid = distributedFFT.finish_subgrid(
            summed_facet,
            [subgrid_config.off0, subgrid_config.off1],
        )
    return approx_subgrid


def prepare_and_split_subgrid(
    distributedFFT, subgrid, Fn, subgrid_offs, facets_config_list
):
    """prepare NAF_NAF"""
    # Prepare subgrid
    prepared_subgrid = distributedFFT.prepare_subgrid(subgrid, subgrid_offs)

    # Extract subgrid facet contributions

    NAF_AFs = {
        off0: distributedFFT.extract_subgrid_contrib_to_facet(
            prepared_subgrid, off0, Fn, axis=0
        )
        for off0 in set(
            facet_config.off0 for facet_config in facets_config_list
        )
    }
    NAF_NAFs = [
        distributedFFT.extract_subgrid_contrib_to_facet(
            NAF_AFs[facet_config.off0], facet_config.off1, Fn, axis=1
        )
        for facet_config in facets_config_list
    ]
    return NAF_NAFs


def accumulate_column(distributedFFT, NAF_NAF, NAF_MNAF, subgrid_off1):
    """update NAF_MNAF"""
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    if NAF_MNAF is None:
        return distributedFFT.add_subgrid_contribution(
            NAF_NAF,
            subgrid_off1,
            axis=1,
        )
    return NAF_MNAF + distributedFFT.add_subgrid_contribution(
        NAF_NAF,
        subgrid_off1,
        axis=1,
    )


def accumulate_facet(
    distributedFFT,
    NAF_MNAF,
    MNAF_BMNAF,
    Fb,
    facet_off1,
    facet_mask1,
    sg_off0,
):
    """update MNAF_BMNAF"""
    if isinstance(facet_mask1, list):
        facet_mask1 = make_mask_from_slice(facet_mask1[0], facet_mask1[1])

    NAF_BMNAF = distributedFFT.finish_facet(
        NAF_MNAF, facet_off1, facet_mask1, Fb, axis=1
    )
    if MNAF_BMNAF is None:
        return distributedFFT.add_subgrid_contribution(
            NAF_BMNAF, sg_off0, axis=0
        )
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    MNAF_BMNAF = MNAF_BMNAF + distributedFFT.add_subgrid_contribution(
        NAF_BMNAF, sg_off0, axis=0
    )
    return MNAF_BMNAF


def finish_facet(distriFFT, MNAF_BMNAF, facet_off0, facet_mask0, Fb):
    """wrapper of finish_facet"""
    if MNAF_BMNAF is not None:
        if isinstance(facet_mask0, list):
            facet_mask0 = make_mask_from_slice(facet_mask0[0], facet_mask0[1])
        approx_facet = distriFFT.finish_facet(
            MNAF_BMNAF,
            facet_off0,
            facet_mask0,
            Fb,
            axis=0,
        )
    else:
        approx_facet = numpy.zeros(
            (distriFFT.yB_size, distriFFT.yB_size), dtype=complex
        )
    return approx_facet


def extract_column(distriFFT, BF_F, Fb_task, subgrid_off0, facet_off1):
    """extract column task"""
    return distriFFT.prepare_facet(
        distriFFT.extract_facet_contrib_to_subgrid(
            BF_F,
            subgrid_off0,
            axis=0,
        ),
        facet_off1,
        Fb_task,
        axis=1,
    )


def make_full_cover_config(N, chunk_size, class_name):
    """Generate a fully covered config list

    :param N: N
    :param chunk_size: facet size or subgrid size
    :param class_name: SubgridConfig or FacetConfig
    :return: config list
    """
    offsets = chunk_size * numpy.arange(int(numpy.ceil(N / chunk_size)))
    border = (offsets + numpy.hstack([offsets[1:], [N + offsets[0]]])) // 2
    config_list = []
    for idx0, off0 in enumerate(offsets):
        for idx1, off1 in enumerate(offsets):
            left0 = (border[idx0 - 1] - off0 + chunk_size // 2) % N
            right0 = border[idx0] - off0 + chunk_size // 2

            left1 = (border[idx1 - 1] - off1 + chunk_size // 2) % N
            right1 = border[idx1] - off1 + chunk_size // 2
            config_list.append(
                class_name(
                    off0,
                    off1,
                    [[slice(left0, right0)], chunk_size],
                    [[slice(left1, right1)], chunk_size],
                )
            )
    return config_list


def make_mask_from_slice(slice_list, mask_size):
    """make mask from sparse_mask result

    :param slice_list: slice list
    :param mask_size: mask size
    :return: mask
    """
    mask = numpy.zeros((mask_size,))
    for sl in slice_list:
        mask[sl] = 1
    return mask
