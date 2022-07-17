# pylint: disable=too-many-arguments,too-few-public-methods,unnecessary-pass
# pylint: disable=consider-using-set-comprehension
"""
Application Programming Interface for Distributed Fourier Transform
"""

import dask
import dask.array
import dask.distributed
import numpy

from src.fourier_transform.fourier_algorithm import (
    make_facet_from_sources,
    make_subgrid_from_sources,
)


class FacetConfig:
    """Facet Configuration Class"""

    def __init__(self, j0, j1, facet_off, facet_B, yB_size):
        """
        Initialize FacetConfig class

        :param facet_off:
        :param facet_B:
        :param yB_size:
        :param j0: j0 index
        :param j1: j1 index

        """
        self.j0 = j0
        self.j1 = j1
        self.facet_off = facet_off
        self.facet_B = facet_B
        self.yBsize = yB_size


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, i0, i1, subgrid_off, subgrid_A, xA_size) -> object:
        """
        Initialize SubgridConfig class

        :param subgrid_off:
        :param subgrid_A:
        :param xA_size:
        :param i0: i0 index
        :param i1: i1 index

        """
        self.i0 = i0
        self.i1 = i1
        self.subgrid_off = subgrid_off
        self.subgrid_A = subgrid_A
        self.xA_size = xA_size


class SwiftlyConfig:
    """Swiftly configuration"""

    def __init__(self):
        pass


def swiftly_forward(
    core_config,
    facets_config_list,
    facets_data,
    subgrid_config_list,
    base_arrays,
):
    """forward sub graph generator

    :param core_config: _description_
    :param facets_config_list: _description_
    :param facets_data: _description_
    :param subgrid_config_list: _description_
    :param base_arrays: _description_
    :yield: _description_
    """
    # persist BF_F
    config_j = from_facets_config_list_get_j0_j1(facets_config_list)
    BF_F_tasks = dask.persist(
        [
            core_config.prepare_facet(
                facets_data[facets_config.j0][facets_config.j1],
                base_arrays.Fb,
                axis=0,
                use_dask=True,
                nout=1,
            )
            for facets_config in facets_config_list
        ]
    )[0]

    config_i = from_subgrid_config_list_get_i0_i1(subgrid_config_list)
    print("config_i:", config_i)
    i0_i1_list = [
        (subgrid_config.i0, subgrid_config.i1)
        for subgrid_config in subgrid_config_list
    ]
    for test in i0_i1_list:
        print(test)
    for i0 in config_i[0]:
        NMBF_BF_task = [
            core_config.prepare_facet(
                core_config.extract_facet_contrib_to_subgrid(
                    BF_F,
                    core_config.subgrid_off[i0],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=0,
                    use_dask=True,
                    nout=1,
                ),
                base_arrays.Fb,
                axis=1,
                use_dask=True,
                nout=1,
            )
            for BF_F in BF_F_tasks
        ]

        for i1 in config_i[1]:
            NMBF_NMBF_tasks = [
                core_config.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    core_config.subgrid_off[i1],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=1,
                    use_dask=True,
                    nout=1,
                )
                for NMBF_BF in NMBF_BF_task
            ]

            # redis
            subgrid_task = dask.delayed(sum_and_finish_subgrid)(
                core_config,
                i0,
                i1,
                zip(config_j[0], config_j[1]),
                NMBF_NMBF_tasks,
                base_arrays,
            )

            # tmp cut no in list
            if (i0, i1) in i0_i1_list:
                yield (i0, i1), subgrid_task


def swiftly_backward(
    core_config,
    facets_config_list,
    subgrid_data,
    subgrid_config_list,
    base_arrays,
):
    """backward sub graph generator

    :param core_config: _description_
    :param facets_config_list: _description_
    :param subgrid_data: _description_
    :param subgrid_config_list: _description_
    :param base_arrays: _description_
    :yield: _description_
    """
    MNAF_BMNAF_tasks = dask.persist(
        [
            dask.delayed(lambda shape: numpy.zeros(shape, dtype="complex128"))(
                (core_config.yP_size, core_config.yB_size)
            )
            for facets_config in facets_config_list
        ]
    )[0]

    config_i = from_subgrid_config_list_get_i0_i1(subgrid_config_list)
    i0_i1_list = [
        (subgrid_config.i0, subgrid_config.i1)
        for subgrid_config in subgrid_config_list
    ]
    j0_j1_list = [
        (facets_config.j0, facets_config.j1)
        for facets_config in facets_config_list
    ]
    for i0 in config_i[0]:

        NAF_MNAF_tasks = [None for facets_config in facets_config_list]

        for i1 in config_i[1]:

            NAF_NAF_tasks = dask.delayed(
                prepare_and_split_subgrid, nout=len(j0_j1_list)
            )(
                core_config,
                base_arrays.Fn,
                j0_j1_list,
                subgrid_data[i0][i1],
            )
            # update NAF_MNAF_tasks
            NAF_MNAF_tasks = [
                dask.delayed(accumulate_column)(
                    core_config,
                    NAF_NAF_task,
                    NAF_MNAF_task,
                    base_arrays.facet_m0_trunc,
                    i1,
                )
                for NAF_NAF_task, NAF_MNAF_task in zip(
                    NAF_NAF_tasks, NAF_MNAF_tasks
                )
            ]

            # task-checker1
            if (i0, i1) in i0_i1_list:
                yield (i0, i1), NAF_MNAF_tasks

        # update MNAF_BMNAF_tasks
        MNAF_BMNAF_tasks = dask.persist(
            [
                dask.delayed(accumulate_facet)(
                    core_config,
                    NAF_MNAF_task,
                    MNAF_BMNAF_task,
                    base_arrays.Fb,
                    base_arrays.facet_m0_trunc,
                    j1,
                    i0,
                    base_arrays,
                )
                for NAF_MNAF_task, MNAF_BMNAF_task, (j0, j1) in zip(
                    NAF_MNAF_tasks, MNAF_BMNAF_tasks, j0_j1_list
                )
            ]
        )[0]

        # task-checker2
        yield (i0, -1), MNAF_BMNAF_tasks
        del NAF_MNAF_tasks

    approx_facet_tasks = [
        core_config.finish_facet(
            MNAF_BMNAF_task,
            base_arrays.facet_B[j0],
            base_arrays.Fb,
            axis=0,
            use_dask=True,
            nout=1,
        )
        for MNAF_BMNAF_task, (j0, j1) in zip(MNAF_BMNAF_tasks, j0_j1_list)
    ]

    yield (-1, -1), approx_facet_tasks


def swiftly_major(
    core_config,
    facets_config_list,
    facets_data,
    obs_subgrid_data,
    obs_subgrid_config_list,
    base_arrays,
):
    """
    forward and backward sub graph generator
    skymodel predict model vis
    res vis = obs vis - model vis
    res vis invert res image

    :param core_config:
    :param facets_config_list:
    :param facets_data:
    :param obs_subgrid_data:
    :param obs_subgrid_config_list:
    :param base_arrays:
    """
    BF_F_tasks = dask.persist(
        [
            core_config.prepare_facet(
                facets_data[facets_config.j0][facets_config.j1],
                base_arrays.Fb,
                axis=0,
                use_dask=True,
                nout=1,
            )
            for facets_config in facets_config_list
        ]
    )[0]

    MNAF_BMNAF_tasks = dask.persist(
        [
            dask.delayed(lambda shape: numpy.zeros(shape, dtype="complex128"))(
                (core_config.yP_size, core_config.yB_size)
            )
            for facets_config in facets_config_list
        ]
    )[0]

    config_i = from_subgrid_config_list_get_i0_i1(obs_subgrid_config_list)
    config_j = from_facets_config_list_get_j0_j1(facets_config_list)
    i0_i1_list = [
        (subgrid_config.i0, subgrid_config.i1)
        for subgrid_config in obs_subgrid_config_list
    ]
    j0_j1_list = [
        (facets_config.j0, facets_config.j1)
        for facets_config in facets_config_list
    ]

    for i0 in config_i[0]:

        NMBF_BF_task = [
            core_config.prepare_facet(
                core_config.extract_facet_contrib_to_subgrid(
                    BF_F,
                    core_config.subgrid_off[i0],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=0,
                    use_dask=True,
                    nout=1,
                ),
                base_arrays.Fb,
                axis=1,
                use_dask=True,
                nout=1,
            )
            for BF_F in BF_F_tasks
        ]

        NAF_MNAF_tasks = [None for facets_config in facets_config_list]

        for i1 in config_i[1]:
            NMBF_NMBF_tasks = [
                core_config.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    core_config.subgrid_off[i1],
                    base_arrays.facet_m0_trunc,
                    base_arrays.Fn,
                    axis=1,
                    use_dask=True,
                    nout=1,
                )
                for NMBF_BF in NMBF_BF_task
            ]

            model_subgrid_data = dask.delayed(sum_and_finish_subgrid)(
                core_config,
                i0,
                i1,
                zip(config_j[0], config_j[1]),
                NMBF_NMBF_tasks,
                base_arrays,
            )

            # subtraction
            residual_subgrid_data = dask.delayed(lambda x, y: x - y)(
                obs_subgrid_data[i0][i1], model_subgrid_data
            )
            # backward
            NAF_NAF_tasks = dask.delayed(
                prepare_and_split_subgrid, nout=len(j0_j1_list)
            )(
                core_config,
                base_arrays.Fn,
                j0_j1_list,
                residual_subgrid_data,
            )
            # update NAF_MNAF_tasks
            NAF_MNAF_tasks = [
                dask.delayed(accumulate_column)(
                    core_config,
                    NAF_NAF_task,
                    NAF_MNAF_task,
                    base_arrays.facet_m0_trunc,
                    i1,
                )
                for NAF_NAF_task, NAF_MNAF_task in zip(
                    NAF_NAF_tasks, NAF_MNAF_tasks
                )
            ]

            # task-checker1
            if (i0, i1) in i0_i1_list:
                yield (i0, i1), NAF_MNAF_tasks

        # update MNAF_BMNAF_tasks
        MNAF_BMNAF_tasks = dask.persist(
            [
                dask.delayed(accumulate_facet)(
                    core_config,
                    NAF_MNAF_task,
                    MNAF_BMNAF_task,
                    base_arrays.Fb,
                    base_arrays.facet_m0_trunc,
                    j1,
                    i0,
                    base_arrays,
                )
                for NAF_MNAF_task, MNAF_BMNAF_task, (j0, j1) in zip(
                    NAF_MNAF_tasks, MNAF_BMNAF_tasks, j0_j1_list
                )
            ]
        )[0]

        # task-checker2
        yield (i0, -1), MNAF_BMNAF_tasks
        del NAF_MNAF_tasks

    residual_facet_tasks = [
        core_config.finish_facet(
            MNAF_BMNAF_task,
            base_arrays.facet_B[j0],
            base_arrays.Fb,
            axis=0,
            use_dask=True,
            nout=1,
        )
        for MNAF_BMNAF_task, (j0, j1) in zip(MNAF_BMNAF_tasks, j0_j1_list)
    ]

    yield (-1, -1), residual_facet_tasks


def from_subgrid_config_list_get_i0_i1(subgrid_config_list):
    """get i0,i1 set"""
    config_i0_list = list(
        set([subgrid_config.i0 for subgrid_config in subgrid_config_list])
    )
    config_i1_list = list(
        set([subgrid_config.i1 for subgrid_config in subgrid_config_list])
    )
    return config_i0_list, config_i1_list


def from_facets_config_list_get_j0_j1(facets_config_data_list):
    """get j0,j1 set"""
    config_j0_list = list(
        set([facets_config.j0 for facets_config in facets_config_data_list])
    )
    config_j1_list = list(
        set([facets_config.j1 for facets_config in facets_config_data_list])
    )
    return config_j0_list, config_j1_list


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
    distr_fft, i0, i1, facet_ixs, NMBF_NMBFs, base_arrays
):
    """sum faect contribution and finsh subgrid"""
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distr_fft.xM_size, distr_fft.xM_size), dtype=complex
    )

    # Add contributions
    for (j0, j1), NMBF_NMBF in zip(facet_ixs, NMBF_NMBFs):
        summed_facet += distr_fft.add_facet_contribution(
            distr_fft.add_facet_contribution(
                NMBF_NMBF, distr_fft.facet_off[j0], axis=0
            ),
            distr_fft.facet_off[j1],
            axis=1,
        )

    # Finish
    return distr_fft.finish_subgrid(
        summed_facet, [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]]
    )


def prepare_and_split_subgrid(distr_fft, Fn, facet_ixs, subgrid):
    """prepare NAF_NAF"""
    # Prepare subgrid
    prepared_subgrid = distr_fft.prepare_subgrid(subgrid)

    # Extract subgrid facet contributions
    NAF_AFs = {
        j0: distr_fft.extract_subgrid_contrib_to_facet(
            prepared_subgrid, distr_fft.facet_off[j0], Fn, axis=0
        )
        for j0 in set(j0 for j0, j1 in facet_ixs)
    }
    NAF_NAFs = [
        distr_fft.extract_subgrid_contrib_to_facet(
            NAF_AFs[j0], distr_fft.facet_off[j1], Fn, axis=1
        )
        for j0, j1 in facet_ixs
    ]
    return NAF_NAFs


def accumulate_column(distr_fft, NAF_NAF, NAF_MNAF, m, i1):
    """update NAF_MNAF"""
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    if NAF_MNAF is None:
        return distr_fft.add_subgrid_contribution(
            NAF_NAF,
            distr_fft.subgrid_off[i1],
            m,
            axis=1,
        )
    return NAF_MNAF + distr_fft.add_subgrid_contribution(
        NAF_NAF,
        distr_fft.subgrid_off[i1],
        m,
        axis=1,
    )


def accumulate_facet(
    distr_fft, NAF_MNAF, MNAF_BMNAF, Fb, m, j1, i0, base_arrays
):
    """update MNAF_BMNAF"""
    NAF_BMNAF = distr_fft.finish_facet(
        NAF_MNAF, base_arrays.facet_B[j1], Fb, axis=1
    )
    if MNAF_BMNAF is None:
        return distr_fft.add_subgrid_contribution(
            NAF_BMNAF, distr_fft.subgrid_off[i0], m, axis=0
        )
    # TODO: add_subgrid_contribution should add
    # directly to NAF_MNAF here at some point.
    MNAF_BMNAF = MNAF_BMNAF + distr_fft.add_subgrid_contribution(
        NAF_BMNAF, distr_fft.subgrid_off[i0], m, axis=0
    )
    return MNAF_BMNAF
