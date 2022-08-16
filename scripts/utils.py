# pylint: disable=too-many-arguments, unused-argument
# pylint: disable=logging-fstring-interpolation

"""
Utility Functions for scripts

We provide functions that help testing and
basic validation of the algorithm.

"""
import logging

import dask
import numpy
from distributed import Lock

from src.utils import single_write_hdf5_task

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)


def sum_and_finish_subgrid(
    distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBFs
):
    """
    Combined function with Sum and Generate Subgrid

    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param facet_ixs: facet index list
    :param i0: i0 index
    :param i1: i1 index
    :param facet_ixs: facet index list
    :param NMBF_NMBFs: list of NMBF_NMBF graph

    :return: i0,i1 index, the shape of approx_subgrid
    """
    # Initialise facet sum
    summed_facet = numpy.zeros(
        (distr_fft.xM_size, distr_fft.xM_size), dtype=complex
    )
    # Add contributions
    # Combined two functions (i.e., Sum and Generate Subgrid)
    #   and run them in serial when use_dask = false
    for (j0, j1), NMBF_NMBF in zip(facet_ixs, NMBF_NMBFs):
        summed_facet += distr_fft.add_facet_contribution(
            distr_fft.add_facet_contribution(
                NMBF_NMBF,
                distr_fft.facet_off[j0],
                axis=0,
                use_dask=False,
                nout=1,
            ),
            distr_fft.facet_off[j1],
            axis=1,
            use_dask=False,
        )

    # Finish
    approx_subgrid = distr_fft.finish_subgrid(
        summed_facet,
        use_dask=False,
        nout=1,
    )

    return approx_subgrid


def wait_for_tasks(work_tasks, timeout=None, return_when="ALL_COMPLETED"):
    """
    Simple function for waiting for tasks to finish.
    Logs completed tasks, and returns list of still-waiting tasks.

    :param work_tasks: task list
    :param timeout: timeout for waiting a task finshed
    :param return_when: return string

    :return: unfinshed task
    """

    # Wait for any task to finish
    dask.distributed.wait(work_tasks, timeout, return_when)

    # Remove finished tasks from work queue, return
    new_work_tasks = []
    for task in work_tasks:
        if task.done():
            log.info(f"Finished:{task.result()}")
        elif task.cancelled():
            log.info("Cancelled")
        else:
            new_work_tasks.append(task)
    return new_work_tasks


def batch_all_i1_NMBF_NMBF(NMBF_BF, distr_fft, base_arrays, batch_i1_list):
    """
    compute NMBF_NMBF with all i1 axis in one task

    :param NMBF_BF: NMBF_BF graph
    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object

    :param i1_batch: batch i1 index list

    :return a batch of NMBF_NMBF
    """
    batch_all = []
    for i1_batch in batch_i1_list:
        NMBF_NMBF_list = []
        for i1 in i1_batch:
            NMBF_NMBF = distr_fft.extract_facet_contrib_to_subgrid(
                NMBF_BF,
                distr_fft.subgrid_off[i1],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                axis=1,
                use_dask=False,
                nout=1,
            )
            NMBF_NMBF_list.append(NMBF_NMBF)
        batch_all.append(NMBF_NMBF_list)
    return batch_all


@dask.delayed
def batch_NMBF_NMBF_sum_finish_subgrid(
    nfacet_ni1_batch_list,
    distr_fft,
    base_arrays,
    facet_ixs,
    i0,
    i1_batch,
    check=False,
):
    """
    Barch Compute NMBF_NMBF and subgrid of i0's in a dask task

    :param NMBF_BF_tasks: NMBF_BF graph
    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param facet_ixs: facet index list
    :param i0: i0 index
    :param i1_batch: batch i1 index list

    :return: approx subgrid index list and shape
    """
    ni1_batch_nfacet_list = list(map(list, zip(*nfacet_ni1_batch_list)))
    approx_subgrid_i0_list = []
    for idx, i1 in enumerate(i1_batch):
        approx_subgrid = sum_and_finish_subgrid(
            distr_fft,
            base_arrays,
            i0,
            i1,
            facet_ixs,
            ni1_batch_nfacet_list[idx],
        )
        if check:
            approx_subgrid_i0_list.append((i0, i1, approx_subgrid))
        else:
            approx_subgrid_i0_list.append((i0, i1, approx_subgrid.shape))
    return approx_subgrid_i0_list


@dask.delayed
def write_approx_subgrid(approx_subgrid_list, base_arrays, hdf5_path):
    """
    write a batch of approx_subgrid to hdf5

    :param approx_subgrid_list: a batch of approx_subgrid
    :param base_arrays: BaseArrays class object
    :param hdf5_path: approx G hdf5 file path

    :return List of i0,i1 and shape of approx_subgrid
    """
    res_list = []
    for i0, i1, approx_subgrid in approx_subgrid_list:
        lock = Lock(hdf5_path)
        lock.acquire()
        _ = single_write_hdf5_task(
            hdf5_path,
            "G_data",
            base_arrays,
            i0,
            i1,
            approx_subgrid,
            use_dask=False,
            nout=1,
        )
        lock.release()
        res_list.append((i0, i1, approx_subgrid.shape))
    return res_list


@dask.delayed
def write_network_transfer_info(path, info):
    """
    write the network transfer info

    :param path: file path
    :param info: info as config1, data_incoming, data_outgoing

    :return info
    """
    lock = Lock(path)
    lock.acquire()
    with open(path, "a+", encoding="utf-8") as f:
        f.write(info + "\n")
    lock.release()
    return info


def human_readable_size(size, decimal_places=3):
    """
    convert human readable bytes size

    :param size: bytes
    :param decimal_places: decimal_places

    :return readable bytes size in str
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def get_and_write_transfer(dask_client, key):
    """get transfer

    :param dask_client: client
    :param key: key for write csv
    """
    dict_outgoing = dask_client.run(
        lambda dask_worker: dask_worker.outgoing_transfer_log
    )
    dict_incoming = dask_client.run(
        lambda dask_worker: dask_worker.incoming_transfer_log
    )

    sum_getitem_incoming = 0.0
    for di_key in dict_incoming.keys():
        for di_key2 in dict_incoming[di_key]:
            # if "getitem" in str(di_key2["keys"]):
            sum_getitem_incoming += di_key2["total"]
    log.info(f"sum_getitem_incoming transfer bytes: {sum_getitem_incoming}")
    sum_getitem_outgoing = 0.0
    for do_key in dict_outgoing.keys():
        for do_key2 in dict_outgoing[do_key]:
            # if "getitem" in str(do_key2["keys"]):
            sum_getitem_outgoing += do_key2["total"]
    log.info(f"sum_getitem_outgoing transfer bytes: {sum_getitem_outgoing}")
    tmp_size_1 = human_readable_size(sum_getitem_incoming)
    tmp_size_2 = human_readable_size(sum_getitem_outgoing)
    write_task = write_network_transfer_info(
        "transfer_info_full_step.txt",
        f"{key},{tmp_size_1},{tmp_size_2}",
    )
    dask_client.compute(write_task, sync=True)
