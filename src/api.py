# pylint: disable=too-many-arguments,too-few-public-methods,unnecessary-pass
# pylint: disable=consider-using-set-comprehension,too-many-instance-attributes
"""
Application Programming Interface for Distributed Fourier Transform
"""

import logging

import dask
import dask.array
import dask.distributed
import numpy
from distributed import Client

from src.api_helper import (
    accumulate_column,
    accumulate_facet,
    extract_column,
    finish_facet,
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

    def __init__(self, j0, j1):
        """
        Initialize FacetConfig class

        :param j0: j0 index
        :param j1: j1 index

        """
        self.j0 = j0
        self.j1 = j1


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, i0, i1):
        """
        Initialize SubgridConfig class

        :param i0: i0 index
        :param i1: i1 index

        """
        self.i0 = i0
        self.i1 = i1


class SwiftlyConfig:
    """Swiftly configuration"""

    def __init__(self, **fundamental_constants):
        self.base_arrays = BaseArrays(**fundamental_constants)
        self.distriFFT = StreamingDistributedFFT(**fundamental_constants)
        self.dask_client = Client.current()


class SwiftlyForward:
    """Swiftly Forward class"""

    def __init__(self, swiftly_config, facet_tasks):
        self.core_config = swiftly_config
        self.facet_tasks = facet_tasks

        self.facets_j_list = [
            [
                (
                    facets_config.j0,
                    facets_config.j1,
                )
                for facets_config, _ in facets_j0
            ]
            for facets_j0 in facet_tasks
        ]

        self.BF_Fs_persist = None

        self.Fb_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.Fb, broadcast=True
        )
        self.facet_m0_trunc_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.facet_m0_trunc, broadcast=True
        )
        self.Fn_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.Fn, broadcast=True
        )
        self.distriFFT_obj_task = self.core_config.dask_client.scatter(
            self.core_config.distriFFT, broadcast=True
        )
        self.base_arrays_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays, broadcast=True
        )

        self.last_i0 = None
        self.NMBF_BFs_i0_persist = None

    def get_subgrid_task(self, subgrid_config):
        """make a subgrid sub graph

        :param subgrid_config: subgrid config
        :return: sub graph
        """
        BF_Fs = self._get_BF_Fs()

        i0 = subgrid_config.i0
        NMBF_BFs_i0 = self.get_NMBF_BFs_i0(i0, BF_Fs)

        i1 = subgrid_config.i1

        return self._gen_subgrid(i0, i1, NMBF_BFs_i0)

    def _gen_subgrid(self, i0, i1, NMBF_BFs_i0):
        """final step for make subgrid

        :param i0: i0
        :param i1: i1
        :param NMBF_BFs_i0: i0-th NMBF_BFs
        :return: subgrid task
        """
        NMBF_NMBF_tasks = [
            [
                self.core_config.distriFFT.extract_facet_contrib_to_subgrid(
                    NMBF_BF,
                    self.core_config.distriFFT.subgrid_off[i1],
                    self.facet_m0_trunc_task,
                    self.Fn_task,
                    axis=1,
                    use_dask=True,
                    nout=1,
                )
                for NMBF_BF in NMBF_BF_j0
            ]
            for NMBF_BF_j0 in NMBF_BFs_i0
        ]
        subgrid_task = dask.delayed(sum_and_finish_subgrid)(
            self.distriFFT_obj_task,
            NMBF_NMBF_tasks,
            self.base_arrays_task,
            i0,
            i1,
            self.facets_j_list,
        )

        return subgrid_task

    def _get_BF_Fs(self):
        """make BF_F prepared facet buffers

        :return: BF_F dict
        """
        if self.BF_Fs_persist is None:
            self.BF_Fs_persist = dask.persist(
                [
                    [
                        self.core_config.distriFFT.prepare_facet(
                            facet_data,
                            self.Fb_task,
                            axis=0,
                            use_dask=True,
                            nout=1,
                        )
                        for facet_config, facet_data in facet_i0
                    ]
                    for facet_i0 in self.facet_tasks
                ]
            )[0]

        return self.BF_Fs_persist

    def get_NMBF_BFs_i0(self, i0, BF_Fs):
        """make i0-th NMBF_BFs

        :param i0: i0
        :param BF_Fs: BF_F task
        :return: i0-th NMBF_BFs dict
        """

        if self.last_i0 != i0:
            if self.NMBF_BFs_i0_persist is not None:
                for NMBF_BF_j0 in self.NMBF_BFs_i0_persist:
                    for NMBF_BF in NMBF_BF_j0:
                        NMBF_BF.cancel()
            self.NMBF_BFs_i0_persist = dask.persist(
                [
                    [
                        dask.delayed(extract_column)(
                            self.distriFFT_obj_task,
                            BF_F,
                            self.Fn_task,
                            self.Fb_task,
                            self.facet_m0_trunc_task,
                            self.core_config.distriFFT.subgrid_off[i0],
                        )
                        for BF_F in BF_F_j0
                    ]
                    for BF_F_j0 in BF_Fs
                ]
            )[0]
            self.last_i0 = i0

        return self.NMBF_BFs_i0_persist


class SwiftlyBackward:
    """Swiftly Backward class"""

    def __init__(self, swiftly_config) -> None:
        self.core_config = swiftly_config

        self.Fb_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.Fb, broadcast=True
        )
        self.facet_m0_trunc_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.facet_m0_trunc, broadcast=True
        )
        self.Fn_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays.Fn, broadcast=True
        )
        self.distriFFT_obj_task = self.core_config.dask_client.scatter(
            self.core_config.distriFFT, broadcast=True
        )
        self.base_arrays_task = self.core_config.dask_client.scatter(
            self.core_config.base_arrays, broadcast=True
        )

        self.facets_j_list = [
            [(j0, j1) for j0 in range(self.core_config.distriFFT.nfacet)]
            for j1 in range(self.core_config.distriFFT.nfacet)
        ]

        self.MNAF_BMNAFs_persist = self._get_MNAF_BMNAFs()
        self.NAF_MNAFs_persist = {}
        self.complete_i0_i1_counter = {}

    def add_new_subgrid_task(self, subgrid_config, new_subgrid_task):
        """add new subgrid task

        :param subgrid_config: subgrid config
        :param new_subgrid_task: new subgrid task
        :return: handle_task
        """
        i0 = subgrid_config.i0
        i1 = subgrid_config.i1

        NAF_NAF_floot = dask.delayed(
            prepare_and_split_subgrid,
            nout=len(self.MNAF_BMNAFs_persist)
            * len(self.MNAF_BMNAFs_persist[0]),
        )(
            self.distriFFT_obj_task,
            self.Fn_task,
            self.facets_j_list,
            new_subgrid_task,
            self.base_arrays_task,
        )
        # split to 2D
        NAF_NAF_tasks = []
        for j0 in range(self.core_config.distriFFT.nfacet):
            NAF_NAF_task_j0 = []
            for j1 in range(self.core_config.distriFFT.nfacet):
                NAF_NAF_task_j0.append(
                    NAF_NAF_floot[j0 * len(self.MNAF_BMNAFs_persist) + j1]
                )
            NAF_NAF_tasks.append(NAF_NAF_task_j0)

        # update i0-th NAF_MNAF_tasks every i1
        new_NAF_MNAFs = self.update_i0_NAF_MNAFs(i0, i1, NAF_NAF_tasks)

        self.update_i0_i1_counter_status(i0, i1)

        task_finished = new_NAF_MNAFs
        # is the last i1 in this i0？，yes, and update MNAF_BMNAF
        if self.is_all_i1_done_in_this_i0(i0):
            task_finished = self.update_MNAF_BMNAFs(i0, new_NAF_MNAFs)
        return task_finished

    def finish(self):
        """finish facet

        :return: approx_facet_tasks
        """

        approx_facet_tasks = [
            [
                dask.delayed(finish_facet)(
                    self.distriFFT_obj_task,
                    self.MNAF_BMNAFs_persist[j0][j1],
                    self.base_arrays_task,
                    self.Fb_task,
                    j0,
                )
                for j1 in range(self.core_config.distriFFT.nfacet)
            ]
            for j0 in range(self.core_config.distriFFT.nfacet)
        ]
        return approx_facet_tasks

    def update_i0_i1_counter_status(self, i0, i1):
        """update i0,i1 counter

        :param i0: i0
        :param i1: i1
        """
        if self.complete_i0_i1_counter.get(i0, None) is None:
            self.complete_i0_i1_counter[i0] = []

        self.complete_i0_i1_counter[i0].append(i1)

    def is_all_i1_done_in_this_i0(self, i0):
        """Is the last i1 in this i0

        :param i0: i0
        :return: True or False
        """
        return (
            self.complete_i0_i1_counter[i0][-1]
            == self.core_config.distriFFT.nsubgrid - 1
        )

    def update_i0_NAF_MNAFs(self, i0, i1, new_NAF_NAF_tasks):
        """update the i0-th NAF_MNAFs

        :param i0: i0
        :param i1: i0
        :param new_NAF_NAF_tasks: new NAF_NAF tasks
        :return: new NAF_MNAF tasks
        """
        old_NAF_MNAFs = self.NAF_MNAFs_persist.get(i0, None)
        if old_NAF_MNAFs is None:
            old_NAF_MNAFs = [
                [None for _ in range(self.core_config.distriFFT.nfacet)]
                for _ in range(self.core_config.distriFFT.nfacet)
            ]
        new_NAF_MNAFs = dask.persist(
            [
                [
                    dask.delayed(accumulate_column)(
                        self.distriFFT_obj_task,
                        new_NAF_NAF_tasks[j0][j1],
                        old_NAF_MNAFs[j0][j1],
                        self.facet_m0_trunc_task,
                        self.core_config.distriFFT.subgrid_off[i1],
                    )
                    for j1 in range(self.core_config.distriFFT.nfacet)
                ]
                for j0 in range(self.core_config.distriFFT.nfacet)
            ]
        )[0]
        self.NAF_MNAFs_persist[i0] = new_NAF_MNAFs
        return new_NAF_MNAFs

    def update_MNAF_BMNAFs(self, i0, new_NAF_MNAFs):
        """update MNAF_BMNAFs

        :param i0: i0
        :param new_NAF_MNAFs: new NAF_MNAF tasks
        :return: updated MNAF_BMNAFs
        """
        self.MNAF_BMNAFs_persist = dask.persist(
            [
                [
                    dask.delayed(accumulate_facet)(
                        self.distriFFT_obj_task,
                        new_NAF_MNAFs[j0][j1],
                        self.MNAF_BMNAFs_persist[j0][j1],
                        self.Fb_task,
                        self.facet_m0_trunc_task,
                        self.base_arrays_task,
                        j1,
                        i0,
                    )
                    for j1 in range(self.core_config.distriFFT.nfacet)
                ]
                for j0 in range(self.core_config.distriFFT.nfacet)
            ]
        )[0]
        return self.MNAF_BMNAFs_persist

    def _get_MNAF_BMNAFs(self):
        MNAF_BMNAFs = dask.persist(
            [
                [
                    dask.delayed(
                        lambda shape: numpy.zeros(shape, dtype="complex128")
                    )(
                        (
                            self.core_config.distriFFT.yP_size,
                            self.core_config.distriFFT.yB_size,
                        )
                    )
                    for _ in range(self.core_config.distriFFT.nfacet)
                ]
                for _ in range(self.core_config.distriFFT.nfacet)
            ]
        )[0]
        return MNAF_BMNAFs


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
