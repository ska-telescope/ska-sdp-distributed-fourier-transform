# pylint: disable=too-many-arguments, unused-argument
# pylint: disable=logging-fstring-interpolation

"""
Utility Functions for scripts

We provide functions that help testing and
basic validation of the algorithm.

"""
import dask
import logging
import numpy
import sys

log = logging.getLogger("fourier-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

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
        base_arrays.subgrid_A[i0],
        base_arrays.subgrid_A[i1],
        use_dask=False,
        nout=1,
    )

    return i0, i1, approx_subgrid.shape


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

@dask.delayed
def batch_NMBF_NMBF_sum_finish_subgrid(
    NMBF_BF_tasks, distr_fft, base_arrays, facet_ixs, i0, i1_batch
):
    """
    Barch Compute NMBF_NMBF and subgrid of i0's in a dask task

    :param NMBF_BF_tasks: NMBF_BF graph
    :param distr_fft: StreamingDistributedFFT class object
    :param base_arrays: BaseArrays class object
    :param facet_ixs: facet index list
    :param i0: i0 index
    :param i1_batch: batch i0 index list

    :return: approx subgrid index list and shape
    """
    approx_subgrid_i0_list = []
    for i1 in i1_batch:
        NMBF_NMBF_list = [
            distr_fft.extract_facet_contrib_to_subgrid(
                NMBF_BF,
                1,
                distr_fft.subgrid_off[i1],
                base_arrays.facet_m0_trunc,
                base_arrays.Fn,
                use_dask=False,
                nout=1,
            )
            for (j0, j1), NMBF_BF in zip(facet_ixs, NMBF_BF_tasks)
        ]
        approx_subgrid_fake_res = sum_and_finish_subgrid(
            distr_fft, base_arrays, i0, i1, facet_ixs, NMBF_NMBF_list
        )
        approx_subgrid_i0_list.append(approx_subgrid_fake_res)
    return approx_subgrid_i0_list

