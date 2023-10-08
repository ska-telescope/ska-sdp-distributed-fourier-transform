# pylint: disable=chained-comparison
"""
Distributed Fourier Transform Module.
Included are a list of base functions that are used across the code.
"""

import numpy


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

    if not isinstance(axis, int) or not isinstance(dims, int):
        raise ValueError(
            "create_slice: axis and dims values have to be integers."
        )

    return tuple(axis_val if i == axis else fill_val for i in range(dims))


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


def generate_masks(N, mask_size, offsets):
    """
    Determine the appropriate masks for cutting out subgrids/facets.
    For each offset in offsets, a mask is generated of size mask_size.
    The mask is centred around the specific offset.

    :param N: Size of the entire image
    :param mask_size: size of the required mask (xA_size or yB_size)
    :param offsets: array of subgrid or facet offsets
                    (subgrid_off or facet_off)

    :return: mask (subgrid_A or facet_B)
    """
    mask = numpy.zeros((len(offsets), mask_size), dtype=int)
    border = (offsets + numpy.hstack([offsets[1:], [N + offsets[0]]])) // 2
    for i, offset in enumerate(offsets):
        left = (border[i - 1] - offset + mask_size // 2) % N
        right = border[i] - offset + mask_size // 2

        if not left >= 0 and right <= mask_size:
            raise ValueError(
                "Mask size not large enough to cover subgrids / facets!"
            )

        mask[i, left:right] = 1

    return mask
