#!/usr/bin/env python
# coding: utf-8
# pylint: disable=unused-argument, too-many-arguments, too-many-locals
"""
Small script for generating hdf5 test files, including FG, G
"""
import argparse
import logging
import os
import sys

import h5py
import numpy
from distributed import Lock

from .dask_wrapper import dask_wrapper, set_up_dask, tear_down_dask
from .fourier_transform import make_subgrid_from_sources

log = logging.getLogger("fourier-data-generater-logger")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


@dask_wrapper
def direct_ft_chunk_work(
    G_2_path, chunk_slice, sources, chunksize, N, use_dask=False, **kwargs
):
    """
    Calculate the value of a chunk of direct fourier transform and
      write to hdf5

    :param G_2_path: the hdf5 file path of G
    :param chunk_slice: slice of hdf5 chunk
    :param sources: sources array
    :param chunksize: size of chunk
    :param N: whole data size

    """

    offs = [s.start - N // 2 + chunksize // 2 for s in chunk_slice]
    chunk_G = make_subgrid_from_sources(sources, N, chunksize, offs)

    # lock
    if use_dask:
        lock = Lock(G_2_path)
        lock.acquire()

    with h5py.File(G_2_path, "r+") as f:
        dataset = f["G_data"]
        dataset[chunk_slice[0], chunk_slice[1]] = chunk_G / (N * N)

    if use_dask:
        lock.release()


def generate_data_hdf5(
    npixel, G_2_path, FG_2_path, chunksize_G, chunksize_FG, client=None
):
    """
    Generate standard data G and FG with hdf5

    :param sparse_ft_class: StreamingDistributedFFT class object
    :param G_2_path: the hdf5 file path of G
    :param FG_2_path: the hdf5 file path of FG
    :param chunksize: size of chunk
    :param client: dask client

    :returns: G_2_path, FG_2_path
    """

    if not os.path.exists(FG_2_path):
        source_count = 10
        sources = numpy.array(
            [
                (
                    numpy.random.rand()
                    * npixel
                    * npixel
                    / numpy.sqrt(source_count)
                    / 2,
                    numpy.random.randint(-npixel // 2, npixel // 2 - 1),
                    numpy.random.randint(-npixel // 2, npixel // 2 - 1),
                )
                for _ in range(source_count)
            ]
        )
        f = h5py.File(FG_2_path, "w")
        FG_dataset = f.create_dataset(
            "FG_data",
            (npixel, npixel),
            dtype="complex128",
            chunks=(chunksize_FG, chunksize_FG),
        )
        # write data point by point
        for i, y, x in sources:
            FG_dataset[int(y) + npixel // 2, int(x) + npixel // 2] += (
                i / npixel / npixel
            )
        f.close()

    if client is None:
        use_dask = False
    else:
        use_dask = True
    if not os.path.exists(G_2_path):
        # create a empty hdf5 file
        f = h5py.File(G_2_path, "w")
        G_dataset = f.create_dataset(
            "G_data",
            (npixel, npixel),
            dtype="complex128",
            chunks=(chunksize_G, chunksize_G),
        )
        chunk_list = []
        for chunk_slice in G_dataset.iter_chunks():
            chunk_list.append(
                direct_ft_chunk_work(
                    G_2_path,
                    chunk_slice,
                    sources,
                    chunksize_G,
                    npixel,
                    use_dask=use_dask,
                    nout=1,
                )
            )
        f.close()

        if use_dask:
            chunk_list = client.compute(chunk_list, sync=True)

    return G_2_path, FG_2_path


def cli_parser():
    """
    Parse command line arguments

    :return: argparse
    """
    parser = argparse.ArgumentParser(
        description="generate G and FG hdf5 file for test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--N", type=int, default=1024, help="hdf5 chunksize for G"
    )

    parser.add_argument(
        "--hdf5_chunksize_G",
        type=int,
        default=256,
        help="hdf5 chunksize for G",
    )

    parser.add_argument(
        "--hdf5_chunksize_FG",
        type=int,
        default=256,
        help="hdf5 chunksize for FG",
    )

    parser.add_argument(
        "--hdf5_prefix", type=str, default="./", help="hdf5 path prefix"
    )

    return parser


def main(args):
    """
    Main function to generate G and FG hdf5 file

    The hdf5 file naming follows the format of G/FG_N_chunksize.h5
    """

    # Fixing seed of numpy random
    numpy.random.seed(123456789)

    scheduler = os.environ.get("DASK_SCHEDULER", None)
    log.info("Scheduler: %s", scheduler)

    dask_client = set_up_dask(scheduler_address=scheduler)
    generate_data_hdf5(
        npixel=args.N,
        G_2_path=f"{args.hdf5_prefix}/\
            G_{args.N}_{args.hdf5_chunksize_G}.h5",
        FG_2_path=f"{args.hdf5_prefix}/\
            FG_{args.N}_{args.hdf5_chunksize_FG}.h5",
        chunksize_G=args.hdf5_chunksize_G,
        chunksize_FG=args.hdf5_chunksize_FG,
        client=dask_client,
    )
    tear_down_dask(dask_client)


if __name__ == "__main__":
    parser_args = cli_parser().parse_args()
    main(parser_args)
