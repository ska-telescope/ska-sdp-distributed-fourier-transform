# pylint: disable=too-many-locals, too-many-arguments, unused-argument
# pylint: disable=too-many-lines, too-many-branches
"""
Utility Functions

We provide functions that help plotting and
basic validation of the algorithm.
"""

import itertools
import logging

import h5py
import numpy
from distributed import Lock
from matplotlib import patches, pylab

from src.fourier_transform.dask_wrapper import dask_wrapper
from src.fourier_transform.fourier_algorithm import (
    coordinates,
    extract_mid,
    fft,
    ifft,
    pad_mid,
    roll_and_extract_mid,
)

log = logging.getLogger("fourier-logger")


# PLOTTING UTILS
def mark_range(
    lbl,
    x0,
    x1=None,
    y0=None,
    y1=None,
    ax=None,
    x_offset=1 / 200,
    linestyle="--",
):
    """
    Helper for marking ranges in a graph.

    :param lbl: graph label
    :param x0: X axis lower limit
    :param x1: X axis upper limit
    :param y0: Y axis lower limit
    :param y1: Y axis upper limit
    :param ax: Axes
    :param x_offset: X offset
    :param linestyle: Linestyle
    """
    if ax is None:
        ax = pylab.gca()
    if y0 is None:
        y0 = ax.get_ylim()[1]
    if y1 is None:
        y1 = ax.get_ylim()[0]
    wdt = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.add_patch(
        patches.PathPatch(
            patches.Path([(x0, y0), (x0, y1)]), linestyle=linestyle
        )
    )
    if x1 is not None:
        ax.add_patch(
            patches.PathPatch(
                patches.Path([(x1, y0), (x1, y1)]), linestyle=linestyle
            )
        )
    else:
        x1 = x0
    if pylab.gca().get_yscale() == "linear":
        lbl_y = (y0 * 7 + y1) / 8
    else:
        # Some type of log scale
        lbl_y = (y0**7 * y1) ** (1 / 8)
    ax.annotate(lbl, (x1 + x_offset * wdt, lbl_y))


def display_plots(x, legend=None, grid=False, xlim=None, fig_name=None):
    """
    Display plots using pylab

    :param x: X values
    :param legend: Legend
    :param grid: If true, construct Grid
    :param xlim: X axis limit (a tuple that contains lower and upper limits)
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
    """
    pylab.clf()
    pylab.rcParams["figure.figsize"] = 16, 8
    pylab.rcParams["image.cmap"] = "viridis"
    if grid:
        pylab.grid()
    if legend is not None:
        pylab.legend(legend)
    if xlim is not None:
        pylab.xlim(xlim)
    pylab.imshow(x)
    pylab.colorbar()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}.png")


def plot_pswf(constants_class, fig_name=None):
    """
    Plot to check that PSWF indeed satisfies intended bounds.

    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
    """
    xN = constants_class.W / constants_class.yN_size / 2
    yN = constants_class.yN_size / 2
    yB = constants_class.yB_size / 2
    xN_size = constants_class.N * constants_class.W / constants_class.yN_size

    pylab.clf()
    pylab.semilogy(
        coordinates(4 * int(xN_size)) * 4 * xN_size / constants_class.N,
        extract_mid(
            numpy.abs(
                ifft(
                    pad_mid(constants_class.pswf, constants_class.N, axis=0),
                    axis=0,
                )
            ),
            4 * int(xN_size),
            axis=0,
        ),
    )
    pylab.legend(["n"])
    mark_range("$x_n$", -xN, xN)
    pylab.xlim(
        -2 * int(xN_size) / constants_class.N,
        (2 * int(xN_size) - 1) / constants_class.N,
    )
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_n.png")

    pylab.clf()
    pylab.semilogy(
        coordinates(constants_class.yN_size) * constants_class.yN_size,
        constants_class.pswf,
    )
    pylab.legend(["$\\mathcal{F}[n]$"])
    mark_range("$y_B$", -yB, yB)
    pylab.xlim(-constants_class.N // 2, constants_class.N // 2 - 1)
    mark_range("$y_n$", -yN, yN)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_fn.png")


def plot_work_terms(constants_class, fig_name=None):
    """
    Calculate actual work terms to use and plot to check them.
    We need both n and b in image space.

    TODO: What exactly are these work terms??
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param fig_name: partial name or prefix (can include path) if figure
                     is saved. If None, pylab.show() is called instead
    """
    pylab.clf()
    pylab.semilogy(
        coordinates(constants_class.xMxN_yP_size)
        / constants_class.yP_size
        * constants_class.xMxN_yP_size,
        constants_class.facet_m0_trunc,
    )
    xM = constants_class.xM_size / 2 / constants_class.N
    mark_range("xM", -xM, xM)
    pylab.grid()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_xm.png")


def _plot_error(err_mean, err_mean_img, fig_name, direction):
    """
    Plot the error terms.

    :param err_mean: Mean of the errors, real part
    :param err_mean_img: Mean of the errors, imaginary part
    :param fig_name: Name of the figure
    :param direction: either "facet_to_subgrid" or "subgrid_to_facet"
                      used for plot naming
    """
    pylab.clf()
    pylab.figure(figsize=(16, 8))
    pylab.imshow(numpy.log(numpy.sqrt(err_mean)) / numpy.log(10))
    pylab.colorbar()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_error_mean_{direction}_2d.png")
    pylab.clf()
    pylab.figure(figsize=(16, 8))
    pylab.imshow(numpy.log(numpy.sqrt(err_mean_img)) / numpy.log(10))
    pylab.colorbar()
    if fig_name is None:
        pylab.show()
    else:
        pylab.savefig(f"{fig_name}_error_mean_image_{direction}_2d.png")


def errors_facet_to_subgrid_2d(
    NMBF_NMBF,
    constants_class,
    subgrid_2,
    to_plot=True,
    fig_name=None,
):
    """
    Calculate the error terms for the 2D facet to subgrid algorithm.

    :param NMBF_NMBF: array of individual facet contributions
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param subgrid_2: 2D numpy array of subgrids
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.

    """
    err_mean = 0
    err_mean_img = 0
    for i0, i1 in itertools.product(
        range(constants_class.nsubgrid), range(constants_class.nsubgrid)
    ):
        approx = numpy.zeros(
            (constants_class.xA_size, constants_class.xA_size), dtype=complex
        )
        approx += NMBF_NMBF[i0, i1]

        err_mean += (
            numpy.abs(approx - subgrid_2[i0, i1]) ** 2
            / constants_class.nsubgrid**2
        )
        err_mean_img += numpy.abs(
            fft(fft(approx - subgrid_2[i0, i1], axis=0), axis=1) ** 2
            / constants_class.nsubgrid**2
        )

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "facet_to_subgrid")


def errors_subgrid_to_facet_2d(
    BMNAF_BMNAF, facet_2, constants_class, to_plot=True, fig_name=None
):
    """
    Calculate the error terms for the 2D subgrid to facet algorithm.

    :param BMNAF_BMNAF: array of individual subgrid contributions
    :param constants_class: BaseArrays class object
                            containing fundamental and derived parameters
    :param facet_2: 2D numpy array of facets
    :param to_plot: run plotting?
    :param fig_name: If given, figures will be saved with this prefix
                     into PNG files. If to_plot is set to False,
                     fig_name doesn't have an effect.
    """
    err_mean = 0
    err_mean_img = 0
    for j0, j1 in itertools.product(
        range(constants_class.nfacet), range(constants_class.nfacet)
    ):
        approx = numpy.zeros(
            (constants_class.yB_size, constants_class.yB_size), dtype=complex
        )
        approx += BMNAF_BMNAF[j0, j1]

        err_mean += (
            numpy.abs(ifft(ifft(approx - facet_2[j0, j1], axis=0), axis=1))
            ** 2
            / constants_class.nfacet**2
        )
        err_mean_img += (
            numpy.abs(approx - facet_2[j0, j1]) ** 2
            / constants_class.nfacet**2
        )

    log.info(
        "RMSE: %s (image: %s)",
        numpy.sqrt(numpy.mean(err_mean)),
        numpy.sqrt(numpy.mean(err_mean_img)),
    )

    if to_plot:
        _plot_error(err_mean, err_mean_img, fig_name, "subgrid_to_facet")


@dask_wrapper
def add_two(one, two, **kwargs):
    """
    Functions for iterative operations to
    accelerate the construction of graphs

    :param one: array or graph
    :param two: array or graph

    :returns: added object
    """
    if one is None:
        return two
    return one + two


def generate_input_data(sparse_ft_class, source_count=10):
    """Generate standard data G and FG
        Memory may not be enough at larger
        scales

    :param sparse_ft_class: sparse_ft_class
    :param source_count: number of sources to add
                         If 0, then don't add sources

    :returns: G,FG
    """
    log.info("\n== Generate input data")

    # adding sources
    if source_count > 0:
        FG_2 = numpy.zeros((sparse_ft_class.N, sparse_ft_class.N))
        sources = [
            (
                numpy.random.randint(
                    -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                ),
                numpy.random.randint(
                    -sparse_ft_class.N // 2, sparse_ft_class.N // 2 - 1
                ),
                numpy.random.rand()
                * sparse_ft_class.N
                * sparse_ft_class.N
                / numpy.sqrt(source_count)
                / 2,
            )
            for _ in range(source_count)
        ]
        for x, y, i in sources:
            FG_2[y + sparse_ft_class.N // 2, x + sparse_ft_class.N // 2] += i
        G_2 = ifft(ifft(FG_2, axis=0), axis=1)

    elif source_count == 0:
        # without sources
        G_2 = (
            numpy.exp(
                2j
                * numpy.pi
                * numpy.random.rand(sparse_ft_class.N, sparse_ft_class.N)
            )
            * numpy.random.rand(sparse_ft_class.N, sparse_ft_class.N)
            / 2
        )
        FG_2 = fft(fft(G_2, axis=0), axis=1)

    else:
        log.info("Invalid number of sources specified.")

    log.info("Mean grid absolute: %s", numpy.mean(numpy.abs(G_2)))
    return G_2, FG_2


@dask_wrapper
def error_task_subgrid_to_facet_2d(approx, true_image, num_true, **kwargs):
    """
    Calculate the error terms for a single 2D subgrid to facet algorithm.

    :param approx: array of individual facets
    :param true_image: true_image of facets
    :param num_true: number of facets

    :returns: err_mean, err_mean_img
    """
    approx_img = numpy.zeros_like(approx) + approx
    err_mean = (
        numpy.abs(ifft(ifft(approx_img - true_image, axis=0), axis=1)) ** 2
        / num_true**2
    )
    err_mean_img = numpy.abs(approx_img - true_image) ** 2 / num_true**2
    return err_mean, err_mean_img


@dask_wrapper
def error_task_facet_to_subgrid_2d(approx, true_image, num_true, **kwargs):
    """
    Calculate the error terms for a single facet to subgrid algorithm.

    :param approx: array of individual subgrid
    :param true_image: true_image of subgrid
    :param num_true: number of subgrid

    :returns: err_mean, err_mean_img
    """
    approx_img = numpy.zeros_like(approx) + approx
    err_mean = numpy.abs(approx_img - true_image) ** 2 / num_true**2
    err_mean_img = numpy.abs(
        fft(fft(approx_img - true_image, axis=0), axis=1) ** 2 / num_true**2
    )
    return err_mean, err_mean_img


@dask_wrapper
def sum_error_task(err_list, **kwargs):
    """
    Summing over error array
    :param err_list: err_mean, err_mean_img list from subgrid or facets
    :returns: sum of err_mean and err_mean_img
    # TODO: need reduce sum
    """
    err_mean = numpy.zeros_like(err_list[0][0])
    err_mean_img = numpy.zeros_like(err_list[0][1])

    for err, err_img in err_list:
        err_mean += err
        err_mean_img += err_img
    return err_mean, err_mean_img


@dask_wrapper
def mean_img_task(err_img, **kwargs):
    """
    the mean of err_mean, err_mean_img

    This function is designed to work with sum_error_task's reduce summation

    :param err_img: err_mean, err_mean_img
    :returns: mean of err_mean and err_mean_img
    """
    return numpy.sqrt(numpy.mean(err_img[0])), numpy.sqrt(
        numpy.mean(err_img[1])
    )


def fundamental_errors(
    approx_what, number_what, true_what, error_task, use_dask
):
    """
    Functions for calculating the error common to facet or subgrid

    :param approx_what: approx subgrid or facets
    :param number_what: number of subgrid or facets
    :param true_what: true subgrid or facets
    :param error_task: error_task_subgrid_to_facet_2d or
            error_task_facet_to_subgrid_2d function
    :returns: mean of err_mean and err_mean_img
    """
    error_task_list = []
    for i0, i1 in itertools.product(range(number_what), range(number_what)):
        tmp_error = error_task(
            approx_what[i0][i1],
            true_what[i0][i1],
            number_what,
            use_dask=use_dask,
            nout=1,
        )
        error_task_list.append(tmp_error)

    error_sum_map = sum_error_task(error_task_list, use_dask=use_dask, nout=1)
    return mean_img_task(error_sum_map, use_dask=use_dask, nout=1)


def errors_facet_to_subgrid_2d_dask(
    approx_subgrid, sparse_ft_class, subgrid_2, use_dask
):
    """
    Functions for calculating the error of approx subgrid

    :param approx_subgrid: approx subgrid
    :param sparse_ft_class: StreamingDistributedFFT class object
    :param subgrid_2: true subgrid
    :returns: mean of err_mean and err_mean_img
    """
    return fundamental_errors(
        approx_subgrid,
        sparse_ft_class.nsubgrid,
        subgrid_2,
        error_task_facet_to_subgrid_2d,
        use_dask,
    )


def errors_subgrid_to_facet_2d_dask(
    approx_facet, facet_2, sparse_ft_class, use_dask
):
    """
    Functions for calculating the error of approx facets

    :param approx_facet: approx facets
    :param sparse_ft_class: StreamingDistributedFFT class object
    :param facet_2: true facets
    :returns: mean of err_mean and err_mean_img
    """
    return fundamental_errors(
        approx_facet,
        sparse_ft_class.nfacet,
        facet_2,
        error_task_subgrid_to_facet_2d,
        use_dask,
    )


@dask_wrapper
def single_write_hdf5_task(
    hdf5_path,
    dataset_name,
    base_arrays,
    idx0,
    idx1,
    block_data,
    use_dask,
    **kwargs,
):
    """
    Single subgrid or facet write hdf5 file task

    :param hdf5_path: approx facets
    :param dataset_name: HDF5 data set name, i.e., G_data, FG_data
    :param block_size: subgrid or facets size
    :param base_arrays: BaseArray class object
    :param idx0: index 0, i.e., i0 or j0
    :param idx1: index 1, i.e., i1 or j1
    :param block_data: subgrid or facets data
    :returns: hdf5 of approx subgrid or facets
    """
    N = base_arrays.N
    if dataset_name == "G_data":
        mask_element_in = base_arrays.subgrid_A
        offset_i = (
            -base_arrays.subgrid_off[idx0],
            -base_arrays.subgrid_off[idx1],
        )
        block_size = base_arrays.xA_size
    elif dataset_name == "FG_data":
        mask_element_in = base_arrays.facet_B
        offset_i = -base_arrays.facet_off[idx0], -base_arrays.facet_off[idx1]
        block_size = base_arrays.yB_size
    else:
        raise ValueError("unsupported dataset_name")
    mask_element = numpy.outer(
        mask_element_in[idx0],
        mask_element_in[idx1],
    )
    block_data = block_data * mask_element
    slicex, slicey = roll_and_extract_mid(
        N, -offset_i[0], block_size
    ), roll_and_extract_mid(N, -offset_i[1], block_size)

    if len(slicex) <= len(slicey):
        iter_what1 = slicex
        iter_what2 = slicey
    else:
        iter_what1 = slicey
        iter_what2 = slicex

    pointx = [0]
    for sl in slicex:
        dt = sl.stop - sl.start
        pointx.append(dt + pointx[-1])

    pointy = [0]
    for sl in slicey:
        dt = sl.stop - sl.start
        pointy.append(dt + pointy[-1])

    # write with Lock
    if use_dask:
        lock = Lock(hdf5_path)
        lock.acquire()
    with h5py.File(hdf5_path, "r+") as f:
        approx_image_dataset = f[dataset_name]

        for i0 in range(len(iter_what1)):
            for i1 in range(len(iter_what2)):
                if len(slicex) <= len(slicey):
                    slice_block_x = slice(pointx[i0], pointx[i0 + 1])
                    slice_block_y = slice(pointy[i1], pointy[i1 + 1])
                    approx_image_dataset[slicex[i0], slicey[i1]] += block_data[
                        slice_block_x, slice_block_y
                    ]  # write it
                else:
                    slice_block_x = slice(pointx[i1], pointx[i1 + 1])
                    slice_block_y = slice(pointy[i0], pointy[i0 + 1])
                    approx_image_dataset[slicex[i1], slicey[i0]] += block_data[
                        slice_block_x, slice_block_y
                    ]
    if use_dask:
        lock.release()
    return hdf5_path


@dask_wrapper
def trim(ls, **kwargs):
    """
    Fetch the first element of a list

    :return: the first item
    """
    return ls[0]


def write_hdf5(
    approx_subgrid,
    approx_facet,
    approx_subgrid_path,
    approx_facet_path,
    base_arrays,
    hdf5_chunksize_G,
    hdf5_chunksize_FG,
    use_dask=True,
):
    """
    Write approx subgrid and facet to hdf5

    :param approx_subgrid: approx subgrid list
    :param approx_facet: approx facet list
    :param approx_subgrid_path: approx subgrid path
    :param approx_facet_path: approx facet path
    :param base_arrays: BaseArrays class object
    :param hdf5_chunksize_G: hdf5 chunk size for G data
    :param hdf5_chunksize_G: hdf5 chunk size for FG data

    :returns: hdf5 path of approx subgrid and facets
    """

    # subgrid
    with h5py.File(approx_subgrid_path, "w") as f:
        f.create_dataset(
            "G_data",
            (base_arrays.N, base_arrays.N),
            dtype="complex128",
            chunks=(hdf5_chunksize_G, hdf5_chunksize_G),
        )
    subgrid_res_list = []
    for i0, i1 in itertools.product(
        range(base_arrays.nsubgrid), range(base_arrays.nsubgrid)
    ):
        res = single_write_hdf5_task(
            approx_subgrid_path,
            "G_data",
            base_arrays,
            i0,
            i1,
            approx_subgrid[i0][i1],
            use_dask=use_dask,
            nout=1,
        )
        subgrid_res_list.append(res)

    # facets
    with h5py.File(approx_facet_path, "w") as f:
        f.create_dataset(
            "FG_data",
            (base_arrays.N, base_arrays.N),
            dtype="complex128",
            chunks=(hdf5_chunksize_FG, hdf5_chunksize_FG),
        )
    facet_res_list = []
    for j0, j1 in itertools.product(
        range(base_arrays.nfacet), range(base_arrays.nfacet)
    ):
        res = single_write_hdf5_task(
            approx_facet_path,
            "FG_data",
            base_arrays,
            j0,
            j1,
            approx_facet[j0][j1],
            use_dask=use_dask,
            nout=1,
        )
        facet_res_list.append(res)

    return trim(subgrid_res_list, use_dask=use_dask, nout=1), trim(
        facet_res_list, use_dask=use_dask, nout=1
    )
