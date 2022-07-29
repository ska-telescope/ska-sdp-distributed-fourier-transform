# pylint: disable=too-many-arguments,too-few-public-methods,unnecessary-pass
# pylint: disable=consider-using-set-comprehension
"""
Application Programming Interface for Distributed Fourier Transform
"""

import logging
import math

import dask
import dask.array
import dask.distributed
import numpy

from src.api_helper import (
    accumulate_column,
    accumulate_facet,
    extract_column,
    generate_mask,
    prepare_and_split_subgrid,
    sum_and_finish_subgrid,
)
from src.fourier_transform.algorithm_parameters import (
    BaseArrays,
    StreamingDistributedFFT,
)

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


class FacetConfig:
    """Facet Configuration Class"""

    def __init__(self, j0, j1, **fundamental_constants):
        """
        Initialize FacetConfig class

        :param j0: j0 index
        :param j1: j1 index
        :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py

        """
        self.j0 = j0
        self.j1 = j1
        self.yB_size = fundamental_constants["yB_size"]

        facet_off = self.yB_size * numpy.arange(
            int(math.ceil(fundamental_constants["N"] / self.yB_size))
        )
        self.facet_off0 = facet_off[j0]
        self.facet_off1 = facet_off[j1]
        # TODO: mask compute
        self.facet_mask0 = generate_mask(
            fundamental_constants["N"], self.yB_size, facet_off
        )[j0]

        self.facet_mask1 = generate_mask(
            fundamental_constants["N"], self.yB_size, facet_off
        )[j1]


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, i0, i1, **fundamental_constants):
        """
        Initialize SubgridConfig class

        :param i0: i0 index
        :param i1: i1 index
        :param fundamental_params: dictionary of fundamental parmeters
                               chosen from swift_configs.py

        """
        self.i0 = i0
        self.i1 = i1
        self.xA_size = fundamental_constants["xA_size"]

        subgrid_off = self.xA_size * numpy.arange(
            int(math.ceil(fundamental_constants["N"] / self.xA_size))
        )
        self.subgrid_off0 = subgrid_off[i0]
        self.subgrid_off1 = subgrid_off[i1]
        # TODO: mask compute
        self.subgrid_mask0 = generate_mask(
            fundamental_constants["N"], self.xA_size, subgrid_off
        )[i0]
        self.subgrid_mask1 = generate_mask(
            fundamental_constants["N"], self.xA_size, subgrid_off
        )[i1]


class SwiftlyConfig:
    """Swiftly configuration"""

    def __init__(self, **fundamental_constants):
        self.base_arrays = BaseArrays(**fundamental_constants)
        self.distriFFT = StreamingDistributedFFT(**fundamental_constants)


def swiftly_forward(
    client,
    core_config,
    facets_config_list,
    facets_data,
    subgrid_config_list,
):
    """forward sub graph generator

    :param core_config: _description_
    :param facets_config_list: _description_
    :param facets_data: _description_
    :param subgrid_config_list: _description_
    :yield: _description_
    """
    # persist BF_F

    Fb_task = client.scatter(core_config.base_arrays.Fb, broadcast=True)
    facet_m0_trunc_task = client.scatter(
        core_config.base_arrays.facet_m0_trunc, broadcast=True
    )
    Fn_task = client.scatter(core_config.base_arrays.Fn, broadcast=True)
    distriFFT_obj_task = client.scatter(core_config.distriFFT, broadcast=True)

    facets_light_j_off = [
        [
            (
                facets_config.j0,
                facets_config.j1,
                facets_config.facet_off0,
                facets_config.facet_off1,
            )
            for facets_config in facets_config_j0
        ]
        for facets_config_j0 in facets_config_list
    ]

    BF_F_tasks = dask.persist(
        [
            [
                core_config.distriFFT.prepare_facet(
                    facets_data[facets_config.j0][facets_config.j1],
                    Fb_task,
                    axis=0,
                    use_dask=True,
                    nout=1,
                )
                for facets_config in facets_config_j0
            ]
            for facets_config_j0 in facets_config_list
        ]
    )[0]

    for subgrid_config_i0 in subgrid_config_list:
        i0 = subgrid_config_i0[0].i0
        NMBF_BF_task = [
            [
                dask.delayed(extract_column)(
                    distriFFT_obj_task,
                    BF_F,
                    Fn_task,
                    Fb_task,
                    facet_m0_trunc_task,
                    subgrid_config_i0[0].subgrid_off0,
                )
                for BF_F in BF_F_j0
            ]
            for BF_F_j0 in BF_F_tasks
        ]

        for subgrid_config in subgrid_config_i0:
            i1 = subgrid_config.i1
            NMBF_NMBF_tasks = [
                [
                    core_config.distriFFT.extract_facet_contrib_to_subgrid(
                        NMBF_BF,
                        subgrid_config.subgrid_off1,
                        facet_m0_trunc_task,
                        Fn_task,
                        axis=1,
                        use_dask=True,
                        nout=1,
                    )
                    for NMBF_BF in NMBF_BF_j0
                ]
                for NMBF_BF_j0 in NMBF_BF_task
            ]

            # re-distributed here
            subgrid_task = dask.delayed(sum_and_finish_subgrid)(
                distriFFT_obj_task,
                NMBF_NMBF_tasks,
                dask.delayed(subgrid_config),
                facets_light_j_off,
            )
            yield "finish subgrid", subgrid_config, (i0, i1), subgrid_task


def swiftly_backward(
    client,
    core_config,
    facets_config_list,
    subgrid_data,
    subgrid_config_list,
):
    """backward sub graph generator

    :param core_config: _description_
    :param facets_config_list: _description_
    :param subgrid_data: _description_
    :param subgrid_config_list: _description_
    :yield: _description_
    """

    Fb_task = client.scatter(core_config.base_arrays.Fb, broadcast=True)
    facet_m0_trunc_task = client.scatter(
        core_config.base_arrays.facet_m0_trunc, broadcast=True
    )
    Fn_task = client.scatter(core_config.base_arrays.Fn, broadcast=True)
    distriFFT_obj_task = client.scatter(core_config.distriFFT, broadcast=True)

    facets_light_j_off = [
        [
            (
                facets_config.j0,
                facets_config.j1,
                facets_config.facet_off0,
                facets_config.facet_off1,
            )
            for facets_config in facets_config_j0
        ]
        for facets_config_j0 in facets_config_list
    ]

    MNAF_BMNAF_tasks = dask.persist(
        [
            [
                dask.delayed(
                    lambda shape: numpy.zeros(shape, dtype="complex128")
                )((core_config.distriFFT.yP_size, facets_config.yB_size))
                for facets_config in facets_config_j0
            ]
            for facets_config_j0 in facets_config_list
        ]
    )[0]

    for subgrid_config_i0 in subgrid_config_list:
        i0 = subgrid_config_i0[0].i0

        NAF_MNAF_tasks = [
            [None for facets_config in facets_config_j0]
            for facets_config_j0 in facets_config_list
        ]

        for subgrid_config in subgrid_config_i0:
            i1 = subgrid_config.i1

            NAF_NAF_floot = dask.delayed(
                prepare_and_split_subgrid,
                nout=len(facets_config_list) * len(facets_config_list[0]),
            )(
                distriFFT_obj_task,
                Fn_task,
                facets_light_j_off,
                subgrid_data[i0][i1],
            )
            # split to 2D
            NAF_NAF_tasks = []
            for facet_config_j0 in facets_config_list:
                NAF_NAF_task_j0 = []
                for facet_config in facet_config_j0:
                    NAF_NAF_task_j0.append(
                        NAF_NAF_floot[
                            facet_config.j0 * len(facet_config_j0)
                            + facet_config.j1
                        ]
                    )
                NAF_NAF_tasks.append(NAF_NAF_task_j0)

            # update NAF_MNAF_tasks
            NAF_MNAF_tasks = [
                [
                    dask.delayed(accumulate_column)(
                        distriFFT_obj_task,
                        NAF_NAF_tasks[facet_config.j0][facet_config.j1],
                        NAF_MNAF_tasks[facet_config.j0][facet_config.j1],
                        facet_m0_trunc_task,
                        subgrid_config.subgrid_off1,
                    )
                    for facet_config in facet_config_j0
                ]
                for facet_config_j0 in facets_config_list
            ]

            # task-checker1
            yield "NAF_MNAF update", subgrid_config, (i0, i1), NAF_MNAF_tasks

        # update MNAF_BMNAF_tasks
        MNAF_BMNAF_tasks = dask.persist(
            [
                [
                    dask.delayed(accumulate_facet)(
                        distriFFT_obj_task,
                        NAF_MNAF_tasks[facet_config.j0][facet_config.j1],
                        MNAF_BMNAF_tasks[facet_config.j0][facet_config.j1],
                        Fb_task,
                        facet_m0_trunc_task,
                        dask.delayed(facet_config.facet_mask1),
                        subgrid_config_i0[0].subgrid_off0,
                    )
                    for facet_config in facet_config_j0
                ]
                for facet_config_j0 in facets_config_list
            ]
        )[0]

        # task-checker2
        yield "MNAF_BMNAF update", -1, (i0, -1), MNAF_BMNAF_tasks
        del NAF_MNAF_tasks

    approx_facet_tasks = [
        [
            core_config.distriFFT.finish_facet(
                MNAF_BMNAF_tasks[facet_config.j0][facet_config.j1],
                dask.delayed(facet_config.facet_mask0),
                Fb_task,
                axis=0,
                use_dask=True,
                nout=1,
            )
            for facet_config in facet_config_j0
        ]
        for facet_config_j0 in facets_config_list
    ]

    yield "finish facet", -1, (-1, -1), approx_facet_tasks


def swiftly_major(
    client,
    core_config,
    facets_config_list,
    facets_data,
    obs_subgrid_config_list,
    obs_subgrid_data,
):
    """
    forward and backward sub graph generator
    skymodel predict model vis
    res vis = obs vis - model vis
    res vis invert res image

    :param core_config:
    :param facets_config_list:
    :param facets_data:
    :param obs_subgrid_config_list:
    :param obs_subgrid_data:

    """

    Fb_task = client.scatter(core_config.base_arrays.Fb, broadcast=True)
    facet_m0_trunc_task = client.scatter(
        core_config.base_arrays.facet_m0_trunc, broadcast=True
    )
    Fn_task = client.scatter(core_config.base_arrays.Fn, broadcast=True)
    distriFFT_obj_task = client.scatter(core_config.distriFFT, broadcast=True)

    facets_light_j_off = [
        [
            (
                facets_config.j0,
                facets_config.j1,
                facets_config.facet_off0,
                facets_config.facet_off1,
            )
            for facets_config in facets_config_j0
        ]
        for facets_config_j0 in facets_config_list
    ]

    BF_F_tasks = dask.persist(
        [
            [
                core_config.distriFFT.prepare_facet(
                    facets_data[facets_config.j0][facets_config.j1],
                    Fb_task,
                    axis=0,
                    use_dask=True,
                    nout=1,
                )
                for facets_config in facets_config_j0
            ]
            for facets_config_j0 in facets_config_list
        ]
    )[0]

    MNAF_BMNAF_tasks = dask.persist(
        [
            [
                dask.delayed(
                    lambda shape: numpy.zeros(shape, dtype="complex128")
                )((core_config.distriFFT.yP_size, facets_config.yB_size))
                for facets_config in facets_config_j0
            ]
            for facets_config_j0 in facets_config_list
        ]
    )[0]

    for subgrid_config_i0 in obs_subgrid_config_list:
        i0 = subgrid_config_i0[0].i0

        NMBF_BF_task = [
            [
                dask.delayed(extract_column)(
                    distriFFT_obj_task,
                    BF_F,
                    Fn_task,
                    Fb_task,
                    facet_m0_trunc_task,
                    subgrid_config_i0[0].subgrid_off0,
                )
                for BF_F in BF_F_j0
            ]
            for BF_F_j0 in BF_F_tasks
        ]

        NAF_MNAF_tasks = [
            [None for facets_config in facets_config_j0]
            for facets_config_j0 in facets_config_list
        ]

        for subgrid_config in subgrid_config_i0:
            i1 = subgrid_config.i1

            NMBF_NMBF_tasks = [
                [
                    core_config.distriFFT.extract_facet_contrib_to_subgrid(
                        NMBF_BF,
                        subgrid_config.subgrid_off1,
                        facet_m0_trunc_task,
                        Fn_task,
                        axis=1,
                        use_dask=True,
                        nout=1,
                    )
                    for NMBF_BF in NMBF_BF_j0
                ]
                for NMBF_BF_j0 in NMBF_BF_task
            ]

            model_subgrid_data = dask.delayed(sum_and_finish_subgrid)(
                distriFFT_obj_task,
                NMBF_NMBF_tasks,
                dask.delayed(subgrid_config),
                facets_light_j_off,
            )

            # subtraction
            residual_subgrid_data = dask.delayed(lambda x, y: x - y)(
                obs_subgrid_data[subgrid_config.i0][subgrid_config.i1],
                model_subgrid_data,
            )
            yield "subtraction", subgrid_config, (i0, i1), [
                [residual_subgrid_data]
            ]

            # backward
            NAF_NAF_floot = dask.delayed(
                prepare_and_split_subgrid,
                nout=len(facets_config_list) * len(facets_config_list[0]),
            )(
                distriFFT_obj_task,
                Fn_task,
                facets_light_j_off,
                residual_subgrid_data,
            )

            # split to 2D
            NAF_NAF_tasks = []
            for j0, facet_config_j0 in enumerate(facets_config_list):
                NAF_NAF_task_j0 = []
                for j1, facet_config in enumerate(facet_config_j0):
                    NAF_NAF_task_j0.append(
                        NAF_NAF_floot[j0 * len(facets_config_list) + j1]
                    )
                NAF_NAF_tasks.append(NAF_NAF_task_j0)

            # update NAF_MNAF_tasks
            NAF_MNAF_tasks = [
                [
                    dask.delayed(accumulate_column)(
                        distriFFT_obj_task,
                        NAF_NAF_tasks[j0][j1],
                        NAF_MNAF_tasks[j0][j1],
                        facet_m0_trunc_task,
                        subgrid_config.subgrid_off1,
                    )
                    for j1, facet_config in enumerate(facet_config_j0)
                ]
                for j0, facet_config_j0 in enumerate(facets_config_list)
            ]
            # task-checker1
            yield "NAF_MNAF update", subgrid_config, (i0, i1), NAF_MNAF_tasks

        # update MNAF_BMNAF_tasks
        MNAF_BMNAF_tasks = dask.persist(
            [
                [
                    dask.delayed(accumulate_facet)(
                        distriFFT_obj_task,
                        NAF_MNAF_tasks[j0][j1],
                        MNAF_BMNAF_tasks[j0][j1],
                        Fb_task,
                        facet_m0_trunc_task,
                        dask.delayed(facet_config.facet_mask1),
                        subgrid_config_i0[0].subgrid_off0,
                    )
                    for j1, facet_config in enumerate(facet_config_j0)
                ]
                for j0, facet_config_j0 in enumerate(facets_config_list)
            ]
        )[0]

        # task-checker2
        yield "NAF_MNAF update", -1, (i0, -1), MNAF_BMNAF_tasks
        del NAF_MNAF_tasks

    residual_facet_tasks = [
        [
            core_config.distriFFT.finish_facet(
                MNAF_BMNAF_tasks[j0][j1],
                dask.delayed(facet_config.facet_mask0),
                Fb_task,
                axis=0,
                use_dask=True,
                nout=1,
            )
            for j1, facet_config in enumerate(facet_config_j0)
        ]
        for j0, facet_config_j0 in enumerate(facets_config_list)
    ]

    yield "finish facet", -1, (-1, -1), residual_facet_tasks


class TaskQueue:
    """Task Queue Class"""

    def __init__(self, max_task):
        """
        Initialize task queue
        :param max_task: max queue size
        """
        self.task_queue = []
        self.meta_queue = []
        self.max_task = max_task
        self.done_tasks = []

    def process(self, msg, coord, task_list):
        """process in queue

        :param msg: msg of sub graph
        :param coord: i0/i1 coord
        :param task_list: task_list
        """

        while len(self.task_queue) >= self.max_task:
            no_done_task = []
            no_done_meta = []
            dask.distributed.wait(
                self.task_queue, return_when="FIRST_COMPLETED"
            )
            for mt, tk in zip(self.meta_queue, self.task_queue):
                if tk.done():
                    self.done_tasks.append((mt, tk))
                else:
                    no_done_task.append(tk)
                    no_done_meta.append(mt)
            self.task_queue = no_done_task
            self.meta_queue = no_done_meta

        handle_task_list_2d = dask.compute(task_list, sync=False)[0]
        idx = 0
        for handle_task_list_1d in handle_task_list_2d:
            for handle_task in handle_task_list_1d:
                self.task_queue.append(handle_task)
                self.meta_queue.append(
                    (
                        idx,
                        len(handle_task_list_1d) * len(handle_task_list_2d),
                        msg,
                        coord,
                    )
                )
                idx += 1

    def empty_done(self):
        """empty tasks which is done"""
        self.done_tasks = []

    def wait_all_done(self):
        """
        wait for all task done
        """

        dask.distributed.wait(self.task_queue)
        for mt, tk in zip(self.meta_queue, self.task_queue):
            if tk.done():
                self.done_tasks.append((mt, tk))
            else:
                raise RuntimeError("some thing error, no complete")
