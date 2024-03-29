# pylint: disable=chained-comparison
"""
Distributed Fourier Transform Module.
Included are a list of base functions that are used across the code.
"""
import itertools

import dask
import h5py
import numpy

from ska_sdp_exec_swiftly.dask_wrapper import dask_wrapper


def create_slice(fill_val, axis_val, dims, axis):
    """
    Create a tuple of length = dims.

    Elements of the tuple:
        fill_val if axis != dim_index;
        axis_val if axis == dim_index,
        where dim_index is each value in range(dims)

    See test for examples.

    :param fill_val: value to use for dimensions where dim != axis
    :param axis_val: value to use for dimensions where dim == axis
    :param dims: length of tuple to be produced
                 (i.e. number of dimensions); int
    :param axis: axis (index) along which axis_val to be used; int

    :return: tuple of length dims
    """
    # pylint: disable=consider-using-generator
    # TODO: pylint's suggestion of using a generator should be investigated
    if not isinstance(axis, int) or not isinstance(dims, int):
        raise ValueError(
            "create_slice: axis and dims values have to be integers."
        )

    return tuple([axis_val if i == axis else fill_val for i in range(dims)])


def broadcast(a, dims, axis):
    """
    Stretch input array to shape determined by the dims and axis values.
    See tests for examples of how the shape of the input array will change
    depending on what dims-axis combination is given

    :param a: input numpy ndarray
    :param dims: dimensions to broadcast ("stretch") input array to; int
    :param axis: axis along which the new dimension(s) should be added; int

    :return: array with new shape
    """
    return a[create_slice(numpy.newaxis, slice(None), dims, axis)]


def pad_mid(a, n, axis):
    """
    Pad an array to a desired size with zeros at a given axis.
    (Surround the middle with zeros till it reaches the given size)

    :param a: numpy array to be padded
    :param n: size to be padded to (desired size)
    :param axis: axis along which to pad

    :return: padded numpy array
    """
    n0 = a.shape[axis]
    if n == n0:
        return a
    pad = create_slice(
        (0, 0),
        (n // 2 - n0 // 2, (n + 1) // 2 - (n0 + 1) // 2),
        len(a.shape),
        axis,
    )
    return numpy.pad(a, pad, mode="constant", constant_values=0.0)


def extract_mid(a, n, axis):
    """
    Extract a section from middle of a map (array) along a given axis.
    This is the reverse operation to pad.

    :param a: numpy array from which to extract
    :param n: size of section
    :param axis: axis along which to extract (int: 0, 1)

    :return: extracted numpy array
    """
    assert n <= a.shape[axis]
    cx = a.shape[axis] // 2
    if n % 2 != 0:
        slc = slice(cx - n // 2, cx + n // 2 + 1)
    else:
        slc = slice(cx - n // 2, cx + n // 2)
    return a[create_slice(slice(None), slc, len(a.shape), axis)]


def fft(a, axis):
    """
    Fourier transformation from image to grid space, along a given axis.

    :param a: numpy array, 1D or 2D (image in `lm` coordinate space)
    :param axis: int; axes over which to calculate

    :return: numpy array (`uv` grid)
    """
    return numpy.fft.fftshift(
        numpy.fft.fft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def ifft(a, axis):
    """
    Fourier transformation from grid to image space, along a given axis.
    (inverse Fourier transform)

    :param a: numpy array, 1D or 2D (`uv` grid to transform)
    :param axis: int; axes over which to calculate

    :return: numpy array (an image in `lm` coordinate space)
    """
    return numpy.fft.fftshift(
        numpy.fft.ifft(numpy.fft.ifftshift(a, axis), axis=axis), axis
    )


def coordinates(n):
    """
    Generate a 1D array with length n,
    which spans [-0.5,0.5] with 0 at position n/2.
    See also docs for numpy.mgrid.

    :param n: length of array to be generated
    :return: 1D numpy array
    """
    n2 = n // 2
    if n % 2 == 0:
        return numpy.mgrid[-n2:n2] / n

    return numpy.mgrid[-n2 : n2 + 1] / n


@dask_wrapper
def ith_subgrid_facet_element(
    true_image, offset_i, true_usable_size, mask_element, axis=(0, 1), **kwargs
):  # pylint: disable=unused-argument
    """
    Calculate a single facet or subgrid element.

    :param true_image: true image, G (1D or 2D)
    :param offset_i: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet
    :param mask_element: an element of subgrid_A or facet_B (masks)
    :param axis: axis (0, 1, or a tuple of both)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    if isinstance(axis, int):
        extracted = extract_mid(
            numpy.roll(true_image, offset_i, axis), true_usable_size, axis
        )

    if isinstance(axis, tuple) and len(axis) == 2:
        extracted = extract_mid(
            extract_mid(
                numpy.roll(true_image, offset_i, axis),
                true_usable_size,
                axis[0],
            ),
            true_usable_size,
            axis[1],
        )

    result = mask_element * extracted
    return result


# TODO: I (GH) tried adding this function as method to the class
#   separate for subgrid and facet, but when calling dask on it
#   the computation becomes extremely slow and my laptop cannot handle it.
#   This suggests that something wasn't right and the dask setup wasn't ideal
#   hence I left these here as a separate function, and not part of the class.
# pylint: disable=too-many-arguments
def make_subgrid_and_facet(
    G,
    FG,
    base_arrays,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param base_arrays: BaseArrays class object
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty(
            (base_arrays.nsubgrid, base_arrays.xA_size), dtype=complex
        )
        facet = numpy.empty(
            (base_arrays.nfacet, base_arrays.yB_size), dtype=complex
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(base_arrays.nsubgrid):
            subgrid[i] = ith_subgrid_facet_element(
                G,
                -base_arrays.subgrid_off[i],
                base_arrays.xA_size,
                base_arrays.subgrid_A[i],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(base_arrays.nfacet):
            facet[j] = ith_subgrid_facet_element(
                FG,
                -base_arrays.facet_off[j],
                base_arrays.yB_size,
                base_arrays.facet_B[j],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty(
            (
                base_arrays.nsubgrid,
                base_arrays.nsubgrid,
                base_arrays.xA_size,
                base_arrays.xA_size,
            ),
            dtype=complex,
        )
        facet = numpy.empty(
            (
                base_arrays.nfacet,
                base_arrays.nfacet,
                base_arrays.yB_size,
                base_arrays.yB_size,
            ),
            dtype=complex,
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(
            range(base_arrays.nsubgrid), range(base_arrays.nsubgrid)
        ):
            subgrid[i0][i1] = ith_subgrid_facet_element(
                G,
                (
                    -base_arrays.subgrid_off[i0],
                    -base_arrays.subgrid_off[i1],
                ),
                base_arrays.xA_size,
                numpy.outer(
                    base_arrays.subgrid_A[i0],
                    base_arrays.subgrid_A[i1],
                ),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(
            range(base_arrays.nfacet), range(base_arrays.nfacet)
        ):
            facet[j0][j1] = ith_subgrid_facet_element(
                FG,
                (
                    -base_arrays.facet_off[j0],
                    -base_arrays.facet_off[j1],
                ),
                base_arrays.yB_size,
                numpy.outer(base_arrays.facet_B[j0], base_arrays.facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


def roll_and_extract_mid(shape, offset, true_usable_size):
    """Calculate the slice of the roll + extract mid method

    :param shape: shape full size data G/FG
    :param offset: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet

    :return: slice list
    """

    centre = shape // 2
    start = centre + offset - true_usable_size // 2

    if true_usable_size % 2 != 0:
        end = centre + offset + true_usable_size // 2 + 1
    else:
        end = centre + offset + true_usable_size // 2

    if end <= 0:
        slice_data = [slice(start + shape, end + shape)]
    elif start < 0 and end > 0:
        slice_data = [slice(0, end), slice(start + shape, shape)]

    elif end <= shape and start >= 0:
        slice_data = [slice(start, end)]
    elif start < shape and end > shape:
        slice_data = [slice(start, shape), slice(0, end - shape)]

    elif start >= shape:
        slice_data = [slice(start - shape, end - shape)]

    else:
        raise ValueError("unsupported slice")

    return slice_data


def roll_and_extract_mid_axis(data, offset, true_usable_size, axis):
    """Calculate the slice of the roll + extract mid along
        with axis method

    :param data: 2D data
    :param offset: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet
    :axis: axis (0 or 1)

    :return: slice list
    """

    slice_list = roll_and_extract_mid(
        data.shape[axis], offset, true_usable_size
    )

    point = [0]
    for slice_item in slice_list:
        delta_slice = slice_item.stop - slice_item.start
        point.append(delta_slice + point[-1])

    # Allocate new data array
    new_shape = list(data.shape)
    new_shape[axis] = true_usable_size
    block_data = numpy.empty(new_shape, dtype=data.dtype)

    # Set data
    for idx0, slice_item in enumerate(slice_list):
        slice_block = slice(point[idx0], point[idx0 + 1])
        out_slices = create_slice(
            slice(None), slice_block, len(data.shape), axis
        )
        in_slices = create_slice(
            slice(None), slice_item, len(data.shape), axis
        )
        block_data[out_slices] = data[in_slices]

    return block_data


# pylint: disable=too-many-locals,unused-argument
@dask_wrapper
def _ith_subgrid_facet_element_from_hdf5(
    hdf5_file, dataset_name, offset_i, base_arrays, idx0, idx1, **kwargs
):
    """
    Calculate a single facet or subgrid element from hdf5 with minimal memory.

    :param hdf5_file: the file path of G/FG hdf5 file
    :param dataset_name: G/FG hdf5 file dataset name
    :param offset_i: ith offset (subgrid or facet)
    :param base_arrays: BaseArrays class object
    :param idx0: index in the axis 0
    :param idx1: index in the axis 1
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    return subgrid or facets element graph
    """
    if dataset_name == "G_data":
        mask_element_in = base_arrays.subgrid_A
        true_usable_size = base_arrays.xA_size
    elif dataset_name == "FG_data":
        mask_element_in = base_arrays.facet_B
        true_usable_size = base_arrays.yB_size
    else:
        raise ValueError("unsupported dataset_name")
    mask_element = numpy.outer(
        mask_element_in[idx0],
        mask_element_in[idx1],
    )
    f = h5py.File(hdf5_file, "r")
    true_image_dataset = f[dataset_name]

    slicex, slicey = roll_and_extract_mid(
        base_arrays.N, -offset_i[0], true_usable_size
    ), roll_and_extract_mid(base_arrays.N, -offset_i[1], true_usable_size)

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

    block_data = numpy.empty(
        (true_usable_size, true_usable_size), dtype="complex128"
    )
    for i0 in range(len(iter_what1)):
        for i1 in range(len(iter_what2)):
            if len(slicex) <= len(slicey):
                slice_block_x = slice(pointx[i0], pointx[i0 + 1])
                slice_block_y = slice(pointy[i1], pointy[i1 + 1])
                block_data[slice_block_x, slice_block_y] = true_image_dataset[
                    slicex[i0], slicey[i1]
                ]
            else:
                slice_block_x = slice(pointx[i1], pointx[i1 + 1])
                slice_block_y = slice(pointy[i0], pointy[i0 + 1])
                block_data[slice_block_x, slice_block_y] = true_image_dataset[
                    slicex[i1], slicey[i0]
                ]

    res = block_data * mask_element
    return res


# pylint: disable=unused-argument
def make_subgrid_and_facet_from_hdf5(
    G,
    FG,
    base_arrays,
    use_dask=True,
):
    """
    Calculate the actual subgrids and facets. Hdf5 & Dask.delayed
    compatible version

    :param G: the path of G hdf5 file
    :param FG: the path of FG hdf5 file
    :param base_arrays: BaseArrays class object
    :param use_dask: run function with dask.delayed or not?
    :return: subgrid and facet graph list
    """

    subgrid = numpy.empty(
        (
            base_arrays.nsubgrid,
            base_arrays.nsubgrid,
        ),
    ).tolist()
    facet = numpy.empty(
        (
            base_arrays.nfacet,
            base_arrays.nfacet,
        ),
    ).tolist()

    for i0, i1 in itertools.product(
        range(base_arrays.nsubgrid), range(base_arrays.nsubgrid)
    ):
        subgrid[i0][i1] = _ith_subgrid_facet_element_from_hdf5(
            G,
            "G_data",
            (
                -base_arrays.subgrid_off[i0],
                -base_arrays.subgrid_off[i1],
            ),
            base_arrays,
            i0,
            i1,
            use_dask=use_dask,
            nout=1,
        )

    for j0, j1 in itertools.product(
        range(base_arrays.nfacet), range(base_arrays.nfacet)
    ):
        facet[j0][j1] = _ith_subgrid_facet_element_from_hdf5(
            FG,
            "FG_data",
            (
                -base_arrays.facet_off[j0],
                -base_arrays.facet_off[j1],
            ),
            base_arrays,
            j0,
            j1,
            use_dask=use_dask,
            nout=1,
        )
    return subgrid, facet


def make_facet_from_sources(
    sources: list[tuple[float, int]],
    image_size: int,
    facet_size: int,
    facet_offsets: list[int],
    facet_masks: list[numpy.ndarray] = None,
):
    """
    Generates a facet from a source list

    This basically boils down to adding pixels on a grid, taking into account
    that coordinates might wrap around. Length of facet_offsets tuple decides
    how many dimensions the result has.

    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :param image_size: All coordinates and offset are
        interpreted as modulo this size
    :param facet_size: Desired size of facet
    :param facet_offsets: Offset tuple of facet mid-point
    :param facet_masks: Mask expressions (optional)
    :returns: Numpy array with facet data
    """

    # Allocate facet
    dims = len(facet_offsets)
    facet = numpy.zeros(dims * [facet_size], dtype=complex)

    # Set indicated pixels on facet
    offs = numpy.array(facet_offsets, dtype=int) - dims * [facet_size // 2]
    for intensity, *coord in sources:

        # Determine position relative to facet centre
        coord = numpy.mod(coord - offs, image_size)

        # Is the source within boundaries?
        if any((coord < 0) | (coord >= facet_size)):
            continue

        # Set pixel
        facet[tuple(coord)] += intensity

    # Apply facet mask
    for axis, mask in enumerate(facet_masks or []):
        facet *= broadcast(numpy.array(mask), dims, axis)

    return facet


def make_subgrid_from_sources(
    sources: list[tuple[float, int]],
    image_size: int,
    subgrid_size: int,
    subgrid_offsets: list[int],
    subgrid_masks: list[numpy.ndarray] = None,
):
    """
    Generates a subgrid from a source list

    This solves a direct Fourier transformation for the given sources.
    Note that in contrast to make_facet_from_sources this can get fairly
    expensive. Length of subgrid_offsets tuple decides how many dimensions
    the result has.

    :param sources: List of (intensity, *coords) tuples, all image
        coordinates integer and relative to image centre
    :param image_size: Image size. Determines grid resolution and
        normalisation.
    :param subgrid_size: Desired size of subgrid
    :param subgrid_offsets: Offset tuple of subgrid mid-point
    :param subgrid_masks: Mask expressions (optional)
    :returns: Numpy array with subgrid data
    """

    # Allocate subgrid
    dims = len(subgrid_offsets)
    subgrid = numpy.zeros(dims * [subgrid_size], dtype=complex)

    # Determine subgrid data via DFT
    uvs = numpy.transpose(
        numpy.mgrid[
            tuple(
                slice(off - subgrid_size // 2, off + (subgrid_size + 1) // 2)
                for off in reversed(subgrid_offsets)
            )
        ][::-1]
    )
    for intensity, *coords in sources:
        norm_int = intensity / image_size**dims
        subgrid += norm_int * numpy.exp(
            (2j * numpy.pi / image_size) * numpy.dot(uvs, coords)
        )

    # Apply subgrid masks
    for axis, mask in enumerate(subgrid_masks or []):
        subgrid *= broadcast(numpy.array(mask), dims, axis)

    return subgrid


def make_subgrid_and_facet_from_sources(sources, base_arrays, use_dask=False):
    """
    Calculate the actual subgrids and facets from a list of specific sources.
    Dask.delayed compatible version
    Currently only works for 2D.

    :param sources: List of source positions
    :param base_arrays: BaseArrays class object
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet) if use_dask=False,
             else, the dask graph of the arrays
    """

    if use_dask:
        facet = [
            [
                dask.delayed(make_facet_from_sources)(
                    sources,
                    base_arrays.N,
                    base_arrays.yB_size,
                    [base_arrays.facet_off[j0], base_arrays.facet_off[j1]],
                    [base_arrays.facet_B[j0], base_arrays.facet_B[j1]],
                )
                for j1 in range(base_arrays.nfacet)
            ]
            for j0 in range(base_arrays.nfacet)
        ]

        subgrid = [
            [
                dask.delayed(make_subgrid_from_sources)(
                    sources,
                    base_arrays.N,
                    base_arrays.xA_size,
                    [base_arrays.subgrid_off[j0], base_arrays.subgrid_off[j1]],
                    [base_arrays.subgrid_A[j0], base_arrays.subgrid_A[j1]],
                )
                for j1 in range(base_arrays.nsubgrid)
            ]
            for j0 in range(base_arrays.nsubgrid)
        ]
    else:

        subgrid = numpy.empty(
            (
                base_arrays.nsubgrid,
                base_arrays.nsubgrid,
                base_arrays.xA_size,
                base_arrays.xA_size,
            ),
            dtype=complex,
        )
        facet = numpy.empty(
            (
                base_arrays.nfacet,
                base_arrays.nfacet,
                base_arrays.yB_size,
                base_arrays.yB_size,
            ),
            dtype=complex,
        )

        for i0, i1 in itertools.product(
            range(base_arrays.nsubgrid), range(base_arrays.nsubgrid)
        ):
            subgrid[i0][i1] = make_subgrid_from_sources(
                sources,
                base_arrays.N,
                base_arrays.xA_size,
                [base_arrays.subgrid_off[i0], base_arrays.subgrid_off[i1]],
                [base_arrays.subgrid_A[i0], base_arrays.subgrid_A[i1]],
            )
        for j0, j1 in itertools.product(
            range(base_arrays.nfacet), range(base_arrays.nfacet)
        ):
            facet[j0][j1] = make_facet_from_sources(
                sources,
                base_arrays.N,
                base_arrays.yB_size,
                [base_arrays.facet_off[j0], base_arrays.facet_off[j1]],
                [base_arrays.facet_B[j0], base_arrays.facet_B[j1]],
            )

    return subgrid, facet
