# pylint: disable=too-many-arguments,too-few-public-methods,unnecessary-pass
# pylint: disable=consider-using-set-comprehension,too-many-instance-attributes
"""
Application Programming Interface for Distributed Fourier Transform
"""
__all__ = [
    "FacetConfig",
    "SubgridConfig",
    "SwiftlyConfig",
    "SwiftlyForward",
    "SwiftlyBackward",
    "make_full_facet_cover",
    "make_full_subgrid_cover",
]

import logging

import dask
import dask.array
import dask.distributed
from distributed import Client

from .api_helper import (
    accumulate_column,
    accumulate_facet,
    extract_column,
    finish_facet,
    make_full_cover_config,
    prepare_and_split_subgrid,
    sum_and_finish_subgrid,
)
from .fourier_transform import BaseArrays, StreamingDistributedFFT

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


class FacetConfig:
    """Facet Configuration Class"""

    def __init__(self, off0, off1, mask0=None, mask1=None):
        """
        Initialize FacetConfig class

        :param off0: off0 index
        :param off1: off1 index
        :param mask0: slices list
        :param mask1: slices list

        """
        self.off0 = off0
        self.off1 = off1
        self.mask0 = mask0
        self.mask1 = mask1


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, off0, off1, mask0=None, mask1=None):
        """
        Initialize SubgridConfig class

        :param off0: off0 index
        :param off1: off1 index
        :param mask0: slices list
        :param mask1: slices list

        """
        self.off0 = off0
        self.off1 = off1
        self.mask0 = mask0
        self.mask1 = mask1


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
        lru_forward=1,
        queue_size=20,
    ):
        self.core_config = swiftly_config
        self.facet_tasks = facet_tasks

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

        self.task_queue = TaskQueue(queue_size)

        self.lru = LRUCache(lru_forward)

    def get_subgrid_task(self, subgrid_config):
        """make a subgrid sub graph

        :param subgrid_config: subgrid config
        :return: sub graph
        """
        BF_Fs = self._get_BF_Fs()

        off0 = subgrid_config.off0
        NMBF_BFs_off0 = self.get_NMBF_BFs_off0(off0, BF_Fs)

        approx_subgrid = self._gen_subgrid(subgrid_config, NMBF_BFs_off0)
        self.task_queue.process(
            [approx_subgrid],
        )
        return approx_subgrid

    def _gen_subgrid(self, subgrid_config, NMBF_BFs_off0):
        """final step for make subgrid

        :param off0: off0
        :param off1: off1
        :param NMBF_BFs_i0: i0-th NMBF_BFs
        :return: subgrid task
        """
        NMBF_NMBF_tasks = [
            self.core_config.distriFFT.extract_facet_contrib_to_subgrid(
                NMBF_BF,
                subgrid_config.off1,
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
            [facet_config for facet_config, _ in self.facet_tasks],
            subgrid_config.mask0,
            subgrid_config.mask1,
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

        NMBF_BFs = self.lru.get(off0)

        if NMBF_BFs is None:
            NMBF_BFs = dask.persist(
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
            self.lru.set(off0, NMBF_BFs)

        return NMBF_BFs


class SwiftlyBackward:
    """Swiftly Backward class"""

    def __init__(
        self,
        swiftly_config,
        facets_config_list,
        lru_backward=1,
        queue_size=20,
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

        self.facets_config_list = facets_config_list

        self.MNAF_BMNAFs_persist = [None for _ in self.facets_config_list]

        self.task_queue = TaskQueue(queue_size)
        self.lru = LRUCache(lru_backward)

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
            self.facets_config_list,
            new_subgrid_task,
        )

        task_finished = self.update_off0_NAF_MNAFs(off0, off1, NAF_NAF_tasks)

        self.task_queue.process(
            task_finished,
        )

        return task_finished

    def finish(self):
        """finish facet

        :return: approx_facet_tasks
        """

        # update the remain updating MNAF_BMNAFs and clean NAF_MNAFs
        for oldest_off0, oldest_NAF_MNAFs in self.lru.pop_all():
            MNAF_BMNAFs_persit = self.update_MNAF_BMNAFs(
                oldest_off0, oldest_NAF_MNAFs
            )
            self.task_queue.process(MNAF_BMNAFs_persit)

        approx_facet_tasks = [
            dask.delayed(finish_facet)(
                self.distriFFT_obj_task,
                MNAF_BMNAF,
                self.Fb_task,
                facet_config.mask0,
            )
            for facet_config, MNAF_BMNAF in zip(
                self.facets_config_list, self.MNAF_BMNAFs_persist
            )
        ]

        self.task_queue.process(approx_facet_tasks)

        self.task_queue.wait_all_done()

        return approx_facet_tasks

    def update_off0_NAF_MNAFs(self, off0, off1, new_NAF_NAF_tasks):
        """update off0 NAF_MNAFs and clean NAF_MNAFs_persist

        :param off0: off0
        :param off1: off1
        :param new_NAF_NAF_tasks: new NAF_NAF
        :return: NAF_MNAFs or list of NAF_MNAFs and update_MNAF_BMNAFs
        """

        old_NAF_NAF_tasks = self.lru.get(off0)
        if old_NAF_NAF_tasks is None:
            old_NAF_NAF_tasks = [None for _ in self.facets_config_list]

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
        oldest_off0, oldest_NAF_MNAFs = self.lru.set(off0, new_NAF_MNAFs)

        if (oldest_off0 is not None) and (oldest_NAF_MNAFs is not None):
            MNAF_BMNAFs_persit = self.update_MNAF_BMNAFs(
                oldest_off0, oldest_NAF_MNAFs
            )
            return_task.append(MNAF_BMNAFs_persit)

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
                    facet_config.mask1,
                    off0,
                )
                for facet_config, new_NAF_MNAF, MNAF_BMNAFs in zip(
                    self.facets_config_list,
                    new_NAF_MNAFs,
                    self.MNAF_BMNAFs_persist,
                )
            ]
        )[0]
        return self.MNAF_BMNAFs_persist


class TaskQueue:
    """Task Queue Class"""

    def __init__(self, max_task):
        """
        Initialize task queue
        :param max_task: max queue size
        """
        self.task_queue = []
        self.max_task = max_task

    def process(self, task_list):
        """process in queue

        :param task_list: task_list
        """
        done_tasks = []
        while len(self.task_queue) >= self.max_task:
            no_done_task = []
            dask.distributed.wait(
                self.task_queue, return_when="FIRST_COMPLETED"
            )
            for tk in self.task_queue:
                if tk.done():
                    done_tasks.append(tk)
                else:
                    no_done_task.append(tk)
            self.task_queue = no_done_task

        if isinstance(task_list[0], list):
            for one_task_list in task_list:
                handle_task_list = dask.compute(one_task_list, sync=False)[0]
                for handle_task in handle_task_list:
                    self.task_queue.append(handle_task)

        else:
            handle_task_list = dask.compute(task_list, sync=False)[0]
            for handle_task in handle_task_list:
                self.task_queue.append(handle_task)

        return done_tasks

    def wait_all_done(self):
        """
        wait for all task done
        """
        done_tasks = []
        dask.distributed.wait(self.task_queue)
        for tk in self.task_queue:
            if tk.done():
                done_tasks.append(tk)
            else:
                raise RuntimeError("some thing error, no complete")
        return done_tasks


class LRUCache:
    """
    LRU cache
    """

    def __init__(self, cache_size):
        """LRU cache

        :param cache_size: lru cache size
        """
        self.cache_size = cache_size
        # The front of the queue is the least frequently used
        # and the end of the queue is the most frequently used
        self.queue = []
        self.hash_map = {}

    def get(self, key):
        """Get a value from the cache and
        update the position of the key in the queue

        :param key: key
        :return: value or None
        """
        res = self.hash_map.get(key, None)
        if res is not None:
            # If this key can be found,
            # place it at the end of the queue
            self.queue.remove(key)
            self.queue.append(key)
        return res

    def set(self, key, value):
        """Add a new value to the cache, update it if it already exists,
        and update the position of the queue, removing the least used
        (queue head) from the cache and queue when the cache is full

        :param key: key
        :param value: value
        :return: least_key key and least_value
        """
        self.hash_map[key] = value

        # If this key is in the queue,
        # update it to the end of the queue
        if key in self.queue:
            self.queue.remove(key)
        self.queue.append(key)

        if len(self.hash_map) <= self.cache_size:
            return None, None
        least_recently_used_key = self.queue.pop(0)
        return least_recently_used_key, self.hash_map.pop(
            least_recently_used_key
        )

    def pop_all(self):
        """pop all value from cache

        :yield: key value
        """
        while len(self.hash_map) > 0:
            least_recently_used_key = self.queue.pop(0)
            least_recently_used_value = self.hash_map.pop(
                least_recently_used_key
            )
            yield least_recently_used_key, least_recently_used_value


def make_full_subgrid_cover(swiftlyconfig):
    """
    make subgrid config list
    """
    return make_full_cover_config(
        swiftlyconfig.distriFFT.N,
        swiftlyconfig.distriFFT.xA_size,
        SubgridConfig,
    )


def make_full_facet_cover(swiftlyconfig):
    """
    make facet config list
    """
    return make_full_cover_config(
        swiftlyconfig.distriFFT.N,
        swiftlyconfig.distriFFT.yB_size,
        FacetConfig,
    )
