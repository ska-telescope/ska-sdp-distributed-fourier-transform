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
    make_mask_from_slice,
    prepare_and_split_subgrid,
    sum_and_finish_subgrid,
)
from .fourier_transform import SwiftlyCore, SwiftlyCoreFunc

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


class FacetConfig:
    """Facet Configuration Class"""

    def __init__(self, off0, off1, size, mask0=None, mask1=None):
        """
        Initialize FacetConfig class

        :param off0: off0 index
        :param off1: off1 index
        :param mask0: slices list
        :param mask1: slices list

        """
        self.off0 = off0
        self.off1 = off1
        self._mask0 = mask0
        self._mask1 = mask1
        self.size = size

    @property
    def mask0(self):
        """Returns vertical facet mask"""
        if isinstance(self._mask0, list):
            return make_mask_from_slice(self._mask0[0], self._mask0[1])
        return self._mask0

    @property
    def mask1(self):
        """Returns horizontal facet mask"""
        if isinstance(self._mask1, list):
            return make_mask_from_slice(self._mask1[0], self._mask1[1])
        return self._mask1


class SubgridConfig:
    """Subgrid Configuration Class"""

    def __init__(self, off0, off1, size, mask0=None, mask1=None):
        """
        Initialize SubgridConfig class

        :param off0: off0 index
        :param off1: off1 index
        :param mask0: slices list
        :param mask1: slices list

        """
        self.off0 = off0
        self.off1 = off1
        self._mask0 = mask0
        self._mask1 = mask1
        self.size = size

    @property
    def mask0(self):
        """Returns vertical subgrid mask"""
        if isinstance(self._mask0, list):
            return make_mask_from_slice(self._mask0[0], self._mask0[1])
        return self._mask0

    @property
    def mask1(self):
        """Returns horizontal subgrid mask"""
        if isinstance(self._mask1, list):
            return make_mask_from_slice(self._mask1[0], self._mask1[1])
        return self._mask1


class SwiftlyConfig:
    """
    Swiftly configuration
    """

    def __init__(
        self,
        W: float,
        fov: float,
        N: int,
        yB_size: int,
        yN_size: int,
        xA_size: int,
        xM_size: int,
        dask_client=None,
        backend="numpy",
        **_other_args,
    ):
        self._W = W
        self._fov = fov
        self._N = N
        self._yB_size = yB_size
        self._yN_size = yN_size
        self._xA_size = xA_size
        self._xM_size = xM_size

        if dask_client is None:
            dask_client = Client.current()
        self.dask_client = dask_client or Client.current()

        # Construct backend routines
        if backend == "numpy":
            self._core = SwiftlyCore(W, N, xM_size, yN_size)
        elif backend == "ska_sdp_func":
            self._core = SwiftlyCoreFunc(W, N, xM_size, yN_size)
        else:
            raise ValueError(f"Unknown SwiFTly backend: {backend}")

        self.core_task = dask.delayed(
            self.dask_client.scatter(self._core, broadcast=True)
        )

    @property
    def image_size(self):
        """
        Size of the entire (virtual) image in pixels
        """
        return self._N

    @property
    def max_facet_size(self):
        """
        Maximum size of a facet in pixels
        """
        return self._yB_size

    @property
    def max_subgrid_size(self):
        """
        Maximum size of a subgrid in pixels
        """
        return self._xA_size

    @property
    def pswf_parameter(self):
        """
        Parameter used for window function

        Needs to be optimised to yield the best trade-off between
        realised accuracy and required facet/subgrid padding.
        """
        return self._W

    @property
    def internal_facet_size(self):
        """
        Size of facet data used internally.

        Includes padding for accuracy / efficiency.
        """
        return self._yN_size

    @property
    def internal_subgrid_size(self):
        """
        Size of subgrid data used internally.

        Includes padding for accuracy / efficiency.
        """
        return self._xM_size

    @property
    def facet_off_step(self):
        """
        Returns the base facet offset.

        All facet offsets must be divisible by this value.
        """
        return self._core.facet_off_step

    @property
    def subgrid_off_step(self):
        """
        Returns the base subgrif offset.

        All subgrid offsets must be divisible by this value.
        """
        return self._core.subgrid_off_step


class SwiftlyForward:
    """Swiftly Forward class"""

    def __init__(
        self,
        swiftly_config,
        facet_tasks,
        lru_forward=1,
        queue_size=20,
        client=None,
    ):
        self.config = swiftly_config
        self.facet_tasks = facet_tasks

        self.BF_Fs_persist = None

        self._client = client or dask.distributed.Client.current()
        self.task_queue = TaskQueue(queue_size, self._client)

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
            self.config.core_task.extract_from_facet(
                NMBF_BF,
                subgrid_config.off1,
                axis=1,
            )
            for NMBF_BF in NMBF_BFs_off0
        ]

        subgrid_task = dask.delayed(sum_and_finish_subgrid)(
            self.config.core_task,
            NMBF_NMBF_tasks,
            [facet_config for facet_config, _ in self.facet_tasks],
            subgrid_config,
        )

        return subgrid_task

    def _get_BF_Fs(self):
        """make BF_F prepared facet buffers

        :return: BF_F dict
        """
        if self.BF_Fs_persist is None:
            self.BF_Fs_persist = self._client.persist(
                [
                    self.config.core_task.prepare_facet(
                        facet_data,
                        facet.off0,
                        axis=0,
                    )
                    for facet, facet_data in self.facet_tasks
                ]
            )

        return self.BF_Fs_persist

    def get_NMBF_BFs_off0(self, off0, BF_Fs):
        """make off0 NMBF_BFs

        :param off0: off0
        :param BF_Fs: BF_F task
        :return: off0 NMBF_BFs dict
        """

        NMBF_BFs = self.lru.get(off0)

        if NMBF_BFs is None:
            NMBF_BFs = self._client.persist(
                [
                    dask.delayed(extract_column)(
                        self.config.core_task,
                        BF_F,
                        off0,
                        facet.off1,
                    )
                    for (facet, _), BF_F in zip(self.facet_tasks, BF_Fs)
                ]
            )
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
        client=None,
    ) -> None:
        self.config = swiftly_config
        self.facets_config_list = facets_config_list

        self.MNAF_BMNAFs_persist = [None for _ in self.facets_config_list]

        self._client = client or dask.distributed.Client.current()
        self.task_queue = TaskQueue(queue_size, self._client)
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
            self.config.core_task,
            new_subgrid_task,
            [off0, off1],
            self.facets_config_list,
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
                self.config.core_task, MNAF_BMNAF, facet_config
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

        new_NAF_MNAFs = self._client.persist(
            [
                dask.delayed(accumulate_column)(
                    self.config.core_task,
                    new_NAF_NAF,
                    old_NAF_MNAF,
                    off1,
                )
                for new_NAF_NAF, old_NAF_MNAF in zip(
                    new_NAF_NAF_tasks, old_NAF_NAF_tasks
                )
            ]
        )

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

        :param off0: x offset of subgrid column
        :param new_NAF_MNAFs: new NAF_MNAF tasks
        :return: updated MNAF_BMNAFs
        """
        self.MNAF_BMNAFs_persist = self._client.persist(
            [
                dask.delayed(accumulate_facet)(
                    self.config.core_task,
                    new_NAF_MNAF,
                    MNAF_BMNAFs,
                    facet_config,
                    off0,
                )
                for facet_config, new_NAF_MNAF, MNAF_BMNAFs in zip(
                    self.facets_config_list,
                    new_NAF_MNAFs,
                    self.MNAF_BMNAFs_persist,
                )
            ]
        )
        return self.MNAF_BMNAFs_persist


class TaskQueue:
    """Task Queue Class"""

    def __init__(self, max_task, client):
        """
        Initialize task queue
        :param max_task: max queue size
        """
        self.task_queue = []
        self.max_task = max_task
        self._client = client

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
                handle_task_list = self._client.compute(
                    one_task_list, sync=False
                )
                for handle_task in handle_task_list:
                    self.task_queue.append(handle_task)

        else:
            handle_task_list = self._client.compute(task_list, sync=False)
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
        swiftlyconfig.image_size,
        swiftlyconfig.max_subgrid_size,
        SubgridConfig,
    )


def make_full_facet_cover(swiftlyconfig):
    """
    make facet config list
    """
    return make_full_cover_config(
        swiftlyconfig.image_size,
        swiftlyconfig.max_facet_size,
        FacetConfig,
    )
