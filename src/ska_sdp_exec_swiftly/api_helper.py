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


def make_subgrid(image_size, sg_config, sources):
    """make subgrid"""

    return make_subgrid_from_sources(
        sources,
        image_size,
        sg_config.size,
        [sg_config.off0, sg_config.off1],
        [sg_config.mask0, sg_config.mask1],
    )


def make_facet(image_size, facet_config, sources):
    """make facet"""

    return make_facet_from_sources(
        sources,
        image_size,
        facet_config.size,
        [facet_config.off0, facet_config.off1],
        [facet_config.mask0, facet_config.mask1],
    )


def check_facet(image_size, facet_config, approx_facet, sources):
    """
    check facet using sources
    """
    # Re-generate facet to compare against

    facet = make_facet(image_size, facet_config, sources)

    # Compare against result
    return numpy.sqrt(numpy.average(numpy.abs(facet - approx_facet) ** 2))


def check_residual(residual_facet):
    """
    check residual image
    """
    return numpy.sqrt(numpy.average(numpy.abs(residual_facet) ** 2))


def check_subgrid(image_size, sg_config, approx_subgrid, sources):
    """
    check subgrid using sources
    """
    # Compare against subgrid (normalised)
    subgrid = make_subgrid_from_sources(
        sources,
        image_size,
        approx_subgrid.shape[0],
        [sg_config.off0, sg_config.off1],
        [sg_config.mask0, sg_config.mask1],
    )
    return numpy.sqrt(numpy.average(numpy.abs(subgrid - approx_subgrid) ** 2))


def sum_and_finish_subgrid(
    distributedFFT,
    NMBF_NMBF_tasks,
    facets_config_list,
    subgrid_config,
):
    """Sum facet contributions to a subgrid and finish it"""

    # Group facets by off1 (i.e. column)
    summed_facet = None
    for off1 in {cfg.off1 for cfg in facets_config_list}:

        # Sum all facet_configs with matching off1
        summed_facet_col = None
        for facet_config, NMBF_NMBF in zip(
            facets_config_list, NMBF_NMBF_tasks
        ):
            if facet_config.off1 != off1:
                continue
            summed_facet_col = distributedFFT.add_facet_contribution(
                NMBF_NMBF, facet_config.off0, axis=0, out=summed_facet_col
            )

        # Add all facets of this column to finished facet
        summed_facet = distributedFFT.add_facet_contribution(
            summed_facet_col, off1, axis=1, out=summed_facet
        )

    # Finish
    result = distributedFFT.finish_subgrid(
        summed_facet,
        [subgrid_config.off0, subgrid_config.off1],
        subgrid_config.size,
    )
    # Apply masks
    if subgrid_config.mask0 is not None:
        result *= subgrid_config.mask0[:, numpy.newaxis]
    if subgrid_config.mask1 is not None:
        result *= subgrid_config.mask1[numpy.newaxis, :]
    return result


def prepare_and_split_subgrid(
    distributedFFT, subgrid, subgrid_offs, facets_config_list
):
    """Prepare subgrid and extract contributions to all facets"""

    # Prepare subgrid
    prepared_subgrid = distributedFFT.prepare_subgrid(subgrid, subgrid_offs)

    # Extract subgrid facet contributions

    NAF_AFs = {
        off0: distributedFFT.extract_subgrid_contrib_to_facet(
            prepared_subgrid, off0, axis=0
        )
        for off0 in set(
            facet_config.off0 for facet_config in facets_config_list
        )
    }
    NAF_NAFs = [
        distributedFFT.extract_subgrid_contrib_to_facet(
            NAF_AFs[facet_config.off0], facet_config.off1, axis=1
        )
        for facet_config in facets_config_list
    ]
    return NAF_NAFs


def accumulate_column(distributedFFT, NAF_NAF, NAF_MNAF, subgrid_off1):
    """
    Sum up subgrid contributions from a subgrid column (i.e. same subgrid
    offset 1) to a facet (NAF_MNAF)

    Note that this is done in-place, so do not to reuse the
    NAF_MNAF parameter after passing it to this function / task!
    """
    return distributedFFT.add_subgrid_contribution(
        NAF_NAF, subgrid_off1, axis=1, out=NAF_MNAF
    )


def accumulate_facet(
    distributedFFT,
    NAF_MNAF,
    MNAF_BMNAF,
    facet_config,
    sg_off0,
):
    """
    Update MNAF_BMNAF

    Note that this is done in-place, so do not to reuse the
    NAF_MNAF parameter after passing it to this function / task!
    """

    NAF_BMNAF = distributedFFT.finish_facet(
        NAF_MNAF,
        facet_config.off1,
        facet_config.size,
        axis=1,
    )
    if facet_config.mask1 is not None:
        NAF_BMNAF *= facet_config.mask1[numpy.newaxis, :]
    return distributedFFT.add_subgrid_contribution(
        NAF_BMNAF, sg_off0, axis=0, out=MNAF_BMNAF
    )


def finish_facet(distriFFT, MNAF_BMNAF, facet_config):
    """wrapper of finish_facet"""
    if MNAF_BMNAF is None:
        return numpy.zeros(
            (distriFFT.yB_size, distriFFT.yB_size), dtype=complex
        )

    approx_facet = distriFFT.finish_facet(
        MNAF_BMNAF,
        facet_config.off0,
        facet_config.size,
        axis=0,
    )
    if facet_config.mask0 is not None:
        approx_facet *= facet_config.mask0[:, numpy.newaxis]
    return approx_facet


def extract_column(distriFFT, BF_F, subgrid_off0, facet_off1):
    """extract column task"""
    return distriFFT.prepare_facet(
        distriFFT.extract_facet_contrib_to_subgrid(
            BF_F,
            subgrid_off0,
            axis=0,
        ),
        facet_off1,
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
                    chunk_size,
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
