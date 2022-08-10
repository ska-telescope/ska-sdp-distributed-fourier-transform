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

    def __init__(self, off0, off1):
        """
        Initialize FacetConfig class

        :param off0: off0 index
        :param off1: off1 index

        """
        self.off0 = off0
        self.off1 = off1


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, off0, off1):
        """
        Initialize SubgridConfig class

        :param off0: off0 index
        :param off1: off1 index

        """
        self.off0 = off0
        self.off1 = off1


class SwiftlyConfig:
    """Swiftly configuration"""

    def __init__(self, **fundamental_constants):
        self.base_arrays = BaseArrays(**fundamental_constants)
        self.distriFFT = StreamingDistributedFFT(**fundamental_constants)
        self.dask_client = Client.current()


class SwiftlyForward:
    """Swiftly Forward class"""

    def __init__(
        self,
        swiftly_config,
        facet_tasks,
        subgrid_mask_off_dict=None,
        lru_forward=1,
    ):
        self.core_config = swiftly_config
        self.facet_tasks = facet_tasks

        self.facets_j_list = [
            (
                facets_config.off0,
                facets_config.off1,
            )
            for facets_config, _ in facet_tasks
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

        self.subgrid_mask_off_dict_task = {}
        if subgrid_mask_off_dict:
            for off in subgrid_mask_off_dict.keys():
                self.subgrid_mask_off_dict_task[
                    off
                ] = self.core_config.dask_client.scatter(
                    subgrid_mask_off_dict[off], broadcast=True
                )

        self.max_off0_NMBF_BFs_persist = lru_forward
        self.NMBF_BFs_off0_persist = {}
        self.NMBF_BFs_off0_queue = []

    def get_subgrid_task(self, subgrid_config):
        """make a subgrid sub graph

        :param subgrid_config: subgrid config
        :return: sub graph
        """
        BF_Fs = self._get_BF_Fs()

        off0 = subgrid_config.off0
        NMBF_BFs_off0 = self.get_NMBF_BFs_off0(off0, BF_Fs)

        off1 = subgrid_config.off1

        return self._gen_subgrid(off0, off1, NMBF_BFs_off0)

    def _gen_subgrid(self, off0, off1, NMBF_BFs_off0):
        """final step for make subgrid

        :param off0: off0
        :param off1: off1
        :param NMBF_BFs_i0: i0-th NMBF_BFs
        :return: subgrid task
        """
        NMBF_NMBF_tasks = [
            self.core_config.distriFFT.extract_facet_contrib_to_subgrid(
                NMBF_BF,
                off1,
                self.facet_m0_trunc_task,
                self.Fn_task,
                axis=1,
                use_dask=True,
                nout=1,
            )
            for NMBF_BF in NMBF_BFs_off0
        ]

        subgrid_task = dask.delayed(sum_and_finish_subgrid)(
            self.distriFFT_obj_task,
            NMBF_NMBF_tasks,
            self.facets_j_list,
            self.subgrid_mask_off_dict_task.get(off0, None),
            self.subgrid_mask_off_dict_task.get(off1, None),
        )

        return subgrid_task

    def _get_BF_Fs(self):
        """make BF_F prepared facet buffers

        :return: BF_F dict
        """
        if self.BF_Fs_persist is None:
            self.BF_Fs_persist = dask.persist(
                [
                    self.core_config.distriFFT.prepare_facet(
                        facet_data,
                        self.Fb_task,
                        axis=0,
                        use_dask=True,
                        nout=1,
                    )
                    for _, facet_data in self.facet_tasks
                ]
            )[0]

        return self.BF_Fs_persist

    def get_NMBF_BFs_off0(self, off0, BF_Fs):
        """make off0 NMBF_BFs

        :param off0: off0
        :param BF_Fs: BF_F task
        :return: off0 NMBF_BFs dict
        """

        if self.NMBF_BFs_off0_persist.get(off0, None) is None:
            self.NMBF_BFs_off0_persist[off0] = dask.persist(
                [
                    dask.delayed(extract_column)(
                        self.distriFFT_obj_task,
                        BF_F,
                        self.Fn_task,
                        self.Fb_task,
                        self.facet_m0_trunc_task,
                        off0,
                    )
                    for BF_F in BF_Fs
                ]
            )[0]
            self.NMBF_BFs_off0_queue.append(off0)

        if len(self.NMBF_BFs_off0_persist) > self.max_off0_NMBF_BFs_persist:
            oldest_off0 = self.NMBF_BFs_off0_queue.pop(0)
            del self.NMBF_BFs_off0_persist[oldest_off0]

        return self.NMBF_BFs_off0_persist[off0]


class SwiftlyBackward:
    """Swiftly Backward class"""

    def __init__(
        self,
        swiftly_config,
        facets_config_list,
        facet_mask_off_dict=None,
        lru_backward=1,
    ) -> None:
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

        self.facets_j_list = [
            (
                facets_config.off0,
                facets_config.off1,
            )
            for facets_config in facets_config_list
        ]

        self.facet_mask_off_dict_task = {}
        if facet_mask_off_dict:
            for off in facet_mask_off_dict.keys():
                self.facet_mask_off_dict_task[
                    off
                ] = self.core_config.dask_client.scatter(
                    facet_mask_off_dict[off], broadcast=True
                )

        self.MNAF_BMNAFs_persist = self._get_MNAF_BMNAFs()

        self.max_off0_NAF_MNAFs_persit = lru_backward
        self.NAF_MNAFs_persist = {}
        self.NAF_MNAFs_queue = []

    def add_new_subgrid_task(self, subgrid_config, new_subgrid_task):
        """add new subgrid task

        :param subgrid_config: subgrid config
        :param new_subgrid_task: new subgrid task
        :return: handle_task
        """
        off0 = subgrid_config.off0
        off1 = subgrid_config.off1

        NAF_NAF_tasks = dask.delayed(
            prepare_and_split_subgrid, nout=len(self.MNAF_BMNAFs_persist)
        )(
            self.distriFFT_obj_task,
            self.Fn_task,
            self.facets_j_list,
            new_subgrid_task,
        )

        task_finished = self.update_off0_NAF_MNAFs(off0, off1, NAF_NAF_tasks)

        return task_finished

    def _update_and_clean_NAF_MNAF(self):
        """
        update and clean all NAF_MNAFs_persist items
        """

        while len(self.NAF_MNAFs_persist) > 0:
            oldest_off0 = self.NAF_MNAFs_queue.pop(0)
            oldest_NAF_MNAFs = self.NAF_MNAFs_persist[oldest_off0]
            update_MNAF_BMNAFs = self.update_MNAF_BMNAFs(
                oldest_off0, oldest_NAF_MNAFs
            )
            del self.NAF_MNAFs_persist[oldest_off0]
            dask.compute(update_MNAF_BMNAFs, sync=False)

    def finish(self):
        """finish facet

        :return: approx_facet_tasks
        """

        # update the remain updating MNAF_BMNAFs and clean NAF_MNAFs
        self._update_and_clean_NAF_MNAF()

        approx_facet_tasks = [
            dask.delayed(finish_facet)(
                self.distriFFT_obj_task,
                MNAF_BMNAF,
                self.Fb_task,
                self.facet_mask_off_dict_task.get(facet_off0, None),
            )
            for (facet_off0, facet_off1), MNAF_BMNAF in zip(
                self.facets_j_list, self.MNAF_BMNAFs_persist
            )
        ]
        return approx_facet_tasks

    def update_off0_NAF_MNAFs(self, off0, off1, new_NAF_NAF_tasks):
        """update off0 NAF_MNAFs and clean NAF_MNAFs_persist

        :param off0: off0
        :param off1: off1
        :param new_NAF_NAF_tasks: new NAF_NAF
        :return: NAF_MNAFs or list of NAF_MNAFs and update_MNAF_BMNAFs
        """

        old_NAF_NAF_tasks = self.NAF_MNAFs_persist.get(off0, None)

        if old_NAF_NAF_tasks is None:
            old_NAF_NAF_tasks = [None for _ in self.facets_j_list]

        new_NAF_MNAFs = dask.persist(
            [
                dask.delayed(accumulate_column)(
                    self.distriFFT_obj_task,
                    new_NAF_NAF,
                    old_NAF_MNAF,
                    self.facet_m0_trunc_task,
                    off1,
                )
                for new_NAF_NAF, old_NAF_MNAF in zip(
                    new_NAF_NAF_tasks, old_NAF_NAF_tasks
                )
            ]
        )[0]

        return_task = [new_NAF_MNAFs]

        self.NAF_MNAFs_persist[off0] = new_NAF_MNAFs

        if not self.NAF_MNAFs_queue:
            self.NAF_MNAFs_queue.append(off0)
        elif self.NAF_MNAFs_queue[-1] != off0:
            self.NAF_MNAFs_queue.append(off0)

        if len(self.NAF_MNAFs_persist) > self.max_off0_NAF_MNAFs_persit:
            oldest_off0 = self.NAF_MNAFs_queue.pop(0)
            oldest_NAF_MNAFs = self.NAF_MNAFs_persist[oldest_off0]
            update_MNAF_BMNAFs = self.update_MNAF_BMNAFs(
                oldest_off0, oldest_NAF_MNAFs
            )
            del self.NAF_MNAFs_persist[oldest_off0]

            return_task.append(update_MNAF_BMNAFs)

        return return_task

    def update_MNAF_BMNAFs(self, off0, new_NAF_MNAFs):
        """update MNAF_BMNAFs

        :param off0: off0
        :param new_NAF_MNAFs: new NAF_MNAF tasks
        :return: updated MNAF_BMNAFs
        """
        self.MNAF_BMNAFs_persist = dask.persist(
            [
                dask.delayed(accumulate_facet)(
                    self.distriFFT_obj_task,
                    new_NAF_MNAF,
                    MNAF_BMNAFs,
                    self.Fb_task,
                    self.facet_m0_trunc_task,
                    self.facet_mask_off_dict_task.get(facet_off1, None),
                    off0,
                )
                for (_, facet_off1), new_NAF_MNAF, MNAF_BMNAFs in zip(
                    self.facets_j_list, new_NAF_MNAFs, self.MNAF_BMNAFs_persist
                )
            ]
        )[0]
        return self.MNAF_BMNAFs_persist

    def _get_MNAF_BMNAFs(self):
        MNAF_BMNAFs = dask.persist(
            [
                dask.delayed(
                    lambda shape: numpy.zeros(shape, dtype="complex128")
                )(
                    (
                        self.core_config.distriFFT.yP_size,
                        self.core_config.distriFFT.yB_size,
                    )
                )
                for _ in self.facets_j_list
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

        if isinstance(task_list[0], list):
            for one_task_list in task_list:
                handle_task_list = dask.compute(one_task_list, sync=False)[0]
                idx = 0

                for handle_task in handle_task_list:
                    self.task_queue.append(handle_task)
                    self.meta_queue.append(
                        (
                            idx,
                            len(handle_task_list),
                            msg,
                            coord,
                        )
                    )
                    idx += 1
        else:
            handle_task_list = dask.compute(task_list, sync=False)[0]
            idx = 0
            for handle_task in handle_task_list:
                self.task_queue.append(handle_task)
                self.meta_queue.append(
                    (
                        idx,
                        len(handle_task_list),
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
