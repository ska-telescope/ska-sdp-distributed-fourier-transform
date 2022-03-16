"""Distributed Fourier Transform Module."""
import itertools

import scipy.special
import scipy.signal
import numpy
import dask.array

from src.fourier_transform.dask_wrapper import dask_wrapper


# TODO: ideally we'd like to merge the 1D functions with their 2D equivalent,
#   which then can be used for both versions


def create_slice(fill_val, axis_val, dims, axis):
    """
    TODO: docstring + tests
    Slice A

    param fill_val: Fill value
    param axis_val: Axis value
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return tuple([axis_val if i == axis else fill_val for i in range(dims)])


def broadcast(a, dims, axis):
    """
    TODO: docstring + tests
    Broadcast A

    param a: A
    param dims: Dimensions
    param axis: Axis

    return:
    """
    return a[create_slice(numpy.newaxis, slice(None), dims, axis)]


def pad_mid(a, n, axis):
    """
    Pad an array to a desired size with zeros at a given axis.
    (Surround the middle with zeros till it reaches the given size)

    param a: numpy array to be padded
    param n: size to be padded to (desired size)
    param axis: axis along which to pad

    return: padded numpy array
    """
    n0 = a.shape[axis]
    if n == n0:
        return a
    pad = create_slice(
        (0, 0), (n // 2 - n0 // 2, (n + 1) // 2 - (n0 + 1) // 2), len(a.shape), axis
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
    TODO: docstring + tests
    1D array which spans [-.5,.5[ with 0 at position N/2"""
    n2 = n // 2
    if n % 2 == 0:
        return numpy.mgrid[-n2:n2] / n
    else:
        return numpy.mgrid[-n2 : n2 + 1] / n


def anti_aliasing_function(shape, m, c):
    """
    TODO: tests
    Compute the prolate spheroidal anti-aliasing function

    See VLA Scientific Memoranda 129, 131, 132
    :param shape: (height, width) pair or just width
    :param m: mode parameter
    :param c: spheroidal parameter
    """

    # One dimensional?
    if len(numpy.array(shape).shape) == 0:

        pswf = scipy.special.pro_ang1(m, m, c, 2 * coordinates(shape))[0]
        pswf[0] = 0  # zap NaN
        return pswf

    # 2D Prolate spheroidal angular function is separable
    return numpy.outer(
        anti_aliasing_function(shape[0], m, c), anti_aliasing_function(shape[1], m, c)
    )


# 1D FOURIER ALGORITHM FUNCTIONS
@dask_wrapper
def _ith_subgrid_facet_element(
    true_image, offset_i, true_usable_size, mask_element, axis=None, **kwargs
):
    """
    :param true_image: true image, G (1D or 2D)
    :param offset_i: ith offset (subgrid or facet)
    :param true_usable_size: xA_size for subgrid, and yB_size for facet
    :param mask_element: an element of subgrid_A or facet_B (masks)
    :param axis: axis (0 or 1)
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
                numpy.roll(true_image, offset_i, axis), true_usable_size, axis[0]
            ),
            true_usable_size,
            axis[1],
        )

    result = mask_element * extracted
    return result


def make_subgrid_and_facet(
    G,
    FG,
    constants_class,
    dims,
    use_dask=False,
):
    """
    Calculate the actual subgrids and facets. Dask.delayed compatible version

    :param G: "ground truth", the actual input data
    :param FG: FFT of input data
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param dims: Dimensions; integer 1 or 2 for 1D or 2D
    :param use_dask: run function with dask.delayed or not?
    :return: tuple of two numpy.ndarray (subgrid, facet)
    """

    if dims == 1:
        subgrid = numpy.empty(
            (constants_class.nsubgrid, constants_class.xA_size), dtype=complex
        )
        facet = numpy.empty(
            (constants_class.nfacet, constants_class.yB_size), dtype=complex
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i in range(constants_class.nsubgrid):
            subgrid[i] = _ith_subgrid_facet_element(
                G,
                -constants_class.subgrid_off[i],
                constants_class.xA_size,
                constants_class.subgrid_A[i],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

        for j in range(constants_class.nfacet):
            facet[j] = _ith_subgrid_facet_element(
                FG,
                -constants_class.facet_off[j],
                constants_class.yB_size,
                constants_class.facet_B[j],
                axis=0,
                use_dask=use_dask,
                nout=1,
            )

    elif dims == 2:
        subgrid = numpy.empty(
            (
                constants_class.nsubgrid,
                constants_class.nsubgrid,
                constants_class.xA_size,
                constants_class.xA_size,
            ),
            dtype=complex,
        )
        facet = numpy.empty(
            (
                constants_class.nfacet,
                constants_class.nfacet,
                constants_class.yB_size,
                constants_class.yB_size,
            ),
            dtype=complex,
        )

        if use_dask:
            subgrid = subgrid.tolist()
            facet = facet.tolist()

        for i0, i1 in itertools.product(
            range(constants_class.nsubgrid), range(constants_class.nsubgrid)
        ):
            subgrid[i0][i1] = _ith_subgrid_facet_element(
                G,
                (-constants_class.subgrid_off[i0], -constants_class.subgrid_off[i1]),
                constants_class.xA_size,
                numpy.outer(
                    constants_class.subgrid_A[i0], constants_class.subgrid_A[i1]
                ),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
        for j0, j1 in itertools.product(
            range(constants_class.nfacet), range(constants_class.nfacet)
        ):
            facet[j0][j1] = _ith_subgrid_facet_element(
                FG,
                (-constants_class.facet_off[j0], -constants_class.facet_off[j1]),
                constants_class.yB_size,
                numpy.outer(constants_class.facet_B[j0], constants_class.facet_B[j1]),
                axis=(0, 1),
                use_dask=use_dask,
                nout=1,
            )
    else:
        raise ValueError("Wrong dimensions. Only 1D and 2D are supported.")

    return subgrid, facet


def make_subgrid_and_facet_dask_array(
    G,
    FG,
    constants_class,
):
    """
    Calculate the actual subgrids and facets. Same as make_subgrid_and_facet
    but implemented using dask.array. Consult that function for a full docstring.
    Only for 1D (2D not yet implemented).

    Returns a dask.array.
    """
    subgrid = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                G,
                -constants_class.subgrid_off[i],
                constants_class.xA_size,
                constants_class.subgrid_A[i],
                axis=0,
            )
            for i in range(constants_class.nsubgrid)
        ],
        chunks=(1, constants_class.xA_size),
    ).astype(complex)

    facet = dask.array.from_array(
        [
            _ith_subgrid_facet_element(
                FG,
                -constants_class.facet_off[j],
                constants_class.yB_size,
                constants_class.facet_B[j],
                axis=0,
            )
            for j in range(constants_class.nfacet)
        ],
        chunks=(1, constants_class.yB_size),
    ).astype(complex)

    return subgrid, facet


@dask_wrapper
def facet_contribution_to_subgrid_1d(
    BjFj,
    offset_i,
    constants_class,
    **kwargs,
):
    """
    Extract the facet contribution to a subgrid for 1D version.

    :param BjFj: Prepared facet data (i.e. multiplied by b, padded to yP_size, Fourier transformed)
    :param offset_i: ith offset value (subgrid)
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

    :return: facet_in_a_subgrid: facet contribution to a subgrid
    """
    MiBjFj = constants_class.facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * constants_class.yP_size // constants_class.N),
        constants_class.xMxN_yP_size,
        axis=0,
    )
    MiBjFj_sum = extract_mid(MiBjFj, constants_class.xM_yP_size, axis=0)
    MiBjFj_sum[: constants_class.xN_yP_size // 2] = (
        MiBjFj_sum[: constants_class.xN_yP_size // 2]
        + MiBjFj[-constants_class.xN_yP_size // 2 :]
    )
    MiBjFj_sum[-constants_class.xN_yP_size // 2 :] = (
        MiBjFj_sum[-constants_class.xN_yP_size // 2 :]
        + MiBjFj[: constants_class.xN_yP_size // 2 :]
    )

    facet_in_a_subgrid = constants_class.Fn * extract_mid(
        fft(MiBjFj_sum, axis=0), constants_class.xM_yN_size, axis=0
    )

    return facet_in_a_subgrid


def facet_contribution_to_subgrid_1d_dask_array(
    BjFj,
    offset_i,
    constants_class,
):
    """
    Extract the facet contribution of a subgrid for 1D version.
    Same as facet_contribution_to_subgrid_1d but implemented using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    MiBjFj = constants_class.facet_m0_trunc * extract_mid(
        numpy.roll(BjFj, -offset_i * constants_class.yP_size // constants_class.N),
        constants_class.xMxN_yP_size,
        axis=0,
    ).rechunk(constants_class.xMxN_yP_size)
    MiBjFj_sum = extract_mid(MiBjFj, constants_class.xM_yP_size, axis=0).rechunk(
        constants_class.xM_yP_size
    )
    MiBjFj_sum[: constants_class.xN_yP_size // 2] += MiBjFj[
        -constants_class.xN_yP_size // 2 :
    ]
    MiBjFj_sum[-constants_class.xN_yP_size // 2 :] += MiBjFj[
        : constants_class.xN_yP_size // 2 :
    ]

    facet_in_a_subgrid = constants_class.Fn * extract_mid(
        fft(MiBjFj_sum, axis=0), constants_class.xM_yN_size, axis=0
    ).rechunk(constants_class.xM_yN_size)

    return facet_in_a_subgrid


@dask_wrapper
def prepare_facet_1d(facet_j, Fb, yP_size, **kwargs):
    """
    :param facet_j: jth facet element
    :param Fb: Fourier transform of grid correction function
    :param yP_size: padded (rough) facet size
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return ifft(pad_mid(facet_j * Fb, yP_size, axis=0), axis=0)  # prepare facet


def facets_to_subgrid_1d(
    facet,
    constants_class,
    dtype,
    use_dask,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.

    :param facet: numpy array of facets
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param dtype: data type
    :param use_dask: use Dask?

    :return: RNjMiBjFj: array of contributions of this facet to different subgrids
    """
    RNjMiBjFj = numpy.empty(
        (constants_class.nsubgrid, constants_class.nfacet, constants_class.xM_yN_size),
        dtype=dtype,
    )

    if use_dask:
        RNjMiBjFj = RNjMiBjFj.tolist()

    for j in range(constants_class.nfacet):
        BjFj = prepare_facet_1d(
            facet[j],
            constants_class.Fb,
            constants_class.yP_size,
            use_dask=use_dask,
            nout=1,
        )
        for i in range(constants_class.nsubgrid):
            RNjMiBjFj[i][j] = facet_contribution_to_subgrid_1d(  # extract subgrid
                BjFj,
                constants_class.subgrid_off[i],
                constants_class,
                use_dask=use_dask,
                nout=1,
            )

    return RNjMiBjFj


def facets_to_subgrid_1d_dask_array(
    facet,
    constants_class,
    dtype,
):
    """
    Facet to subgrid 1D algorithm. Returns redistributed subgrid data.
    Same as facets_to_subgrid_1d but implemented using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    RNjMiBjFj = dask.array.from_array(
        [
            [
                facet_contribution_to_subgrid_1d_dask_array(
                    ifft(
                        pad_mid(
                            facet[i] * constants_class.Fb,
                            constants_class.yP_size,
                            axis=0,
                        ).rechunk(constants_class.yP_size),
                        axis=0,
                    ),  # prepare facet
                    constants_class.subgrid_off[j],
                    constants_class,
                )
                for i in range(constants_class.nfacet)
            ]
            for j in range(constants_class.nsubgrid)
        ],
        chunks=(1, 1, constants_class.xM_yN_size),
    ).astype(dtype)
    return RNjMiBjFj


@dask_wrapper
def _add_padded_value(nmbf, facet_off_j, xM_size, N, **kwargs):
    """
    :param nmbf: a single element of the subgrid array calculated from
                 facets by facets_to_subgrid_1d
    :param facet_off_j: jth facet offset
    :param xM_size: padded (rough) subgrid size
    :param N: total image size
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return numpy.roll(pad_mid(nmbf, xM_size, axis=0), facet_off_j * xM_size // N)


def reconstruct_subgrid_1d(nmbfs, constants_class, use_dask):
    """
    Reconstruct the subgrid array calculated by facets_to_subgrid_1d.
    TODO: why are we doing this? how is the result different from the result of facets_to_subgrid_1d?

    Note: compared to reconstruct_facet_1d, which does the same but for facets,
    the order in which we perform calculations is different:
    Here we first do the sum, and then cut from xM_size to xA_size,
    hence approx starts as a larger sized array;
    In reconstruct_facet_1d we apply fft then extract_mid to cut down from size yP_size to yB_size,
    finally we do the sum (which results in approx), hence approx is already defined with the smaller size.

    :param nmbfs: subgrid array calculated from facets by facets_to_subgrid_1d
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: use Dask?

    :return: approximate subgrid
    """
    approx_subgrid = numpy.ndarray(
        (constants_class.nsubgrid, constants_class.xA_size), dtype=complex
    )
    if use_dask:
        approx_subgrid = approx_subgrid.tolist()
        approx = numpy.zeros(
            (constants_class.nsubgrid, constants_class.xM_size), dtype=complex
        )
        approx = approx.tolist()
        for i in range(constants_class.nsubgrid):
            for j in range(constants_class.nfacet):
                approx[i] = approx[i] + _add_padded_value(
                    nmbfs[i][j],
                    constants_class.facet_off[j],
                    constants_class.xM_size,
                    constants_class.N,
                    use_dask=use_dask,
                    nout=1,
                )
            # TODO: Here we used dask array in order to avoid complications of ifft, but this is not optimal.
            approx_array = dask.array.from_delayed(
                approx[i], (constants_class.xM_size,), dtype=complex
            )
            approx_subgrid[i] = constants_class.subgrid_A[i] * extract_mid(
                ifft(approx_array, axis=0), constants_class.xA_size, axis=0
            )
    else:
        for i in range(constants_class.nsubgrid):
            approx = numpy.zeros(constants_class.xM_size, dtype=complex)
            for j in range(constants_class.nfacet):
                approx = approx + _add_padded_value(
                    nmbfs[i, j],
                    constants_class.facet_off[j],
                    constants_class.xM_size,
                    constants_class.N,
                )
            approx_subgrid[i, :] = constants_class.subgrid_A[i] * extract_mid(
                ifft(approx, axis=0), constants_class.xA_size, axis=0
            )

    return approx_subgrid


def reconstruct_subgrid_1d_dask_array(nmbfs, constants_class):
    """
    Reconstruct the subgrid array calculated by facets_to_subgrid_1d.
    Same as reconstruct_subgrid_1d but using dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    approx = dask.array.from_array(
        numpy.zeros((constants_class.nsubgrid, constants_class.xM_size), dtype=complex),
        chunks=(1, constants_class.xM_size),
    )
    approx_subgrid = dask.array.from_array(
        numpy.ndarray(
            (constants_class.nsubgrid, constants_class.xA_size), dtype=complex
        ),
        chunks=(1, constants_class.xA_size),
    )
    for i in range(constants_class.nsubgrid):
        for j in range(constants_class.nfacet):
            padded = pad_mid(nmbfs[i, j, :], constants_class.xM_size, axis=0).rechunk(
                constants_class.xM_size
            )
            approx[i, :] += numpy.roll(
                padded,
                constants_class.facet_off[j]
                * constants_class.xM_size
                // constants_class.N,
            )

        approx_subgrid[i, :] = constants_class.subgrid_A[i] * extract_mid(
            ifft(approx[i], axis=0), constants_class.xA_size, axis=0
        ).rechunk(constants_class.xA_size)

    return approx_subgrid


@dask_wrapper
def _calculate_fns_term(subgrid_ith, facet_off_jth, constants_class, **kwargs):
    """
    :param subgrid_ith: ith subgrid element
    :param facet_off_jth: jth facet offset
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1
    """
    return constants_class.Fn * extract_mid(
        numpy.roll(
            fft(pad_mid(subgrid_ith, constants_class.xM_size, axis=0), axis=0),
            -facet_off_jth * constants_class.xM_size // constants_class.N,
        ),
        constants_class.xM_yN_size,
        axis=0,
    )


def subgrid_to_facet_1d(subgrid, constants_class, use_dask):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.

    :param subgrid: numpy array of subgrids
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: use Dask?

    :return: distributed facet array determined from input subgrid array
    """
    FNjSi = numpy.empty(
        (constants_class.nsubgrid, constants_class.nfacet, constants_class.xM_yN_size),
        dtype=complex,
    )

    if use_dask:
        FNjSi = FNjSi.tolist()

    for i in range(constants_class.nsubgrid):
        for j in range(constants_class.nfacet):
            FNjSi[i][j] = _calculate_fns_term(
                subgrid[i],
                constants_class.facet_off[j],
                constants_class,
                use_dask=use_dask,
                nout=1,
            )

    return FNjSi


def subgrid_to_facet_1d_dask_array(subgrid, constants_class):
    """
    Subgrid to facet algorithm. Returns redistributed facet data.
    Same as subgrid_to_facet_1d but implemented with dask.array
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    FNjSi = dask.array.from_array(
        [
            [
                extract_mid(
                    numpy.roll(
                        fft(
                            pad_mid(
                                subgrid[j], constants_class.xM_size, axis=0
                            ).rechunk(constants_class.xM_size),
                            axis=0,
                        ),
                        -constants_class.facet_off[i]
                        * constants_class.xM_size
                        // constants_class.N,
                    ),
                    constants_class.xM_yN_size,
                    axis=0,
                )
                for i in range(constants_class.nfacet)
            ]
            for j in range(constants_class.nsubgrid)
        ]
    )
    distributed_facet = constants_class.Fn * FNjSi

    return distributed_facet


@dask_wrapper
def add_subgrid_contribution_1d(
    nafs_ij,
    subgrid_off_i,
    constants_class,
    **kwargs,
):
    """
    Add subgrid contribution to a single facet.

    :param nafs_ij: redistributed facet array [i:j] element
    :param subgrid_off_i: subgrid offset [i]th element
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: subgrid contribution
    """
    NjSi = numpy.zeros(constants_class.xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, constants_class.xM_yP_size, axis=0)
    NjSi_mid[:] = ifft(
        pad_mid(nafs_ij, constants_class.xM_yP_size, axis=0), axis=0
    )  # updates NjSi via reference!
    NjSi[-constants_class.xN_yP_size // 2 :] = NjSi_mid[
        : constants_class.xN_yP_size // 2
    ]
    NjSi[: constants_class.xN_yP_size // 2 :] = NjSi_mid[
        -constants_class.xN_yP_size // 2 :
    ]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(
                constants_class.facet_m0_trunc * NjSi, constants_class.yP_size, axis=0
            ),
            subgrid_off_i * constants_class.yP_size // constants_class.N,
        ),
        axis=0,
    )
    subgrid_contrib = extract_mid(FMiNjSi, constants_class.yB_size, axis=0)
    return subgrid_contrib


def add_subgrid_contribution_1d_dask_array(
    nafs_ij,
    subgrid_off_i,
    constants_class,
):
    """
    Add subgrid contribution to a single facet.
    Same as add_subgrid_contribution_1d but implemented with dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    NjSi = numpy.zeros(constants_class.xMxN_yP_size, dtype=complex)
    NjSi_mid = extract_mid(NjSi, constants_class.xM_yP_size, axis=0)
    NjSi_mid[:] = ifft(
        pad_mid(nafs_ij, constants_class.xM_yP_size, axis=0).rechunk(
            constants_class.xM_yP_size
        ),
        axis=0,
    )  # updates NjSi via reference!
    NjSi[-constants_class.xN_yP_size // 2 :] = NjSi_mid[
        : constants_class.xN_yP_size // 2
    ]
    NjSi[: constants_class.xN_yP_size // 2 :] = NjSi_mid[
        -constants_class.xN_yP_size // 2 :
    ]
    FMiNjSi = fft(
        numpy.roll(
            pad_mid(
                constants_class.facet_m0_trunc * NjSi, constants_class.yP_size, axis=0
            ),
            subgrid_off_i * constants_class.yP_size // constants_class.N,
        ),
        axis=0,
    )
    subgrid_contrib = extract_mid(FMiNjSi, constants_class.yB_size, axis=0)
    return subgrid_contrib


def reconstruct_facet_1d(
    nafs,
    constants_class,
    use_dask=False,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d

    :param nafs: redistributed facet array calculated from facets by subgrid_to_facet_1d
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param use_dask: whether to run with dask.delayed or not

    :return: approximate facet
    """
    # Note: compared to reconstruct_subgrid_1d, which does the same but for subgrids,
    # the order in which we perform calculations is different:
    # here, we apply fft then extract_mid to cut down from size yP_size to yB_size,
    # finally we do the sum (which results in approx);
    # In reconstruct_subgrid_1d we first do the sum, and then cut from xM_size to xA_size,
    # hence approx starts as a larger sized array in that case.

    approx_facet = numpy.ndarray(
        (constants_class.nfacet, constants_class.yB_size), dtype=complex
    )
    if use_dask:
        approx_facet = approx_facet.tolist()
        approx = numpy.zeros(
            (constants_class.nfacet, constants_class.yB_size), dtype=complex
        )
        approx = approx.tolist()
        for j in range(constants_class.nfacet):
            for i in range(constants_class.nsubgrid):
                approx[j] = approx[j] + add_subgrid_contribution_1d(
                    nafs[i][j],
                    constants_class.subgrid_off[i],
                    constants_class,
                    use_dask=use_dask,
                    nout=1,
                )
            approx_facet[j] = (
                approx[j] * constants_class.Fb * constants_class.facet_B[j]
            )
    else:
        for j in range(constants_class.nfacet):
            approx = numpy.zeros(constants_class.yB_size, dtype=complex)
            for i in range(constants_class.nsubgrid):
                approx = approx + add_subgrid_contribution_1d(
                    nafs[i, j],
                    constants_class.subgrid_off[i],
                    constants_class,
                )
            approx_facet[j, :] = (
                approx * constants_class.Fb * constants_class.facet_B[j]
            )

    return approx_facet


def reconstruct_facet_1d_dask_array(
    nafs,
    constants_class,
):
    """
    Reconstruct the facet array calculated by subgrid_to_facet_1d.
    Same as reconstruct_facet_1d but implemented with dask.array.
    Consult that function for a full docstring.

    Returns a dask.array.
    """
    approx = dask.array.from_array(
        numpy.zeros((constants_class.nfacet, constants_class.yB_size), dtype=complex),
        chunks=(1, constants_class.yB_size),
    )  # why is this not yP_size? for the subgrid version it uses xM_size
    approx_facet = dask.array.from_array(
        numpy.ndarray((constants_class.nfacet, constants_class.yB_size), dtype=complex),
        chunks=(1, constants_class.yB_size),
    )
    for j in range(constants_class.nfacet):
        for i in range(constants_class.nsubgrid):
            approx[j, :] += add_subgrid_contribution_1d_dask_array(
                nafs[i, j],
                constants_class.subgrid_off[i],
                constants_class,
            )
        approx_facet[j, :] = approx[j] * constants_class.Fb * constants_class.facet_B[j]

    return approx_facet


# 2D FOURIER ALGORITHM FUNCTIONS (facet to subgrid)
@dask_wrapper
def prepare_facet(facet, axis, Fb, yP_size, **kwargs):
    """

    :param facet: Facet
    :param axis: Axis
    :param Fb:
    :param yP_size: Facet size, padded for m convolution (internal)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: BF
    """
    BF = pad_mid(facet * broadcast(Fb, len(facet.shape), axis), yP_size, axis)
    BF = ifft(BF, axis)
    return BF


@dask_wrapper
def extract_subgrid(
    BF,
    axis,
    subgrid_off_i,
    constants_class,
    **kwargs,
):
    """
    Extract the facet contribution of a subgrid.
    TODO: maybe rename function to reflect this definition
        See discussion at https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/merge_requests/4#note_825003275

    :param BF:
    :param axis: Axis
    :param subgrid_off_i:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return:
    """
    dims = len(BF.shape)
    BF_mid = extract_mid(
        numpy.roll(
            BF, -subgrid_off_i * constants_class.yP_size // constants_class.N, axis
        ),
        constants_class.xMxN_yP_size,
        axis,
    )
    MBF = broadcast(constants_class.facet_m0_trunc, dims, axis) * BF_mid
    MBF_sum = numpy.array(extract_mid(MBF, constants_class.xM_yP_size, axis))
    xN_yP_size = constants_class.xMxN_yP_size - constants_class.xM_yP_size
    # [:xN_yP_size//2] / [-xN_yP_size//2:] for axis, [:] otherwise
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    MBF_sum[slc1] += MBF[slc2]
    MBF_sum[slc2] += MBF[slc1]

    return broadcast(constants_class.Fn, len(BF.shape), axis) * extract_mid(
        fft(MBF_sum, axis), constants_class.xM_yN_size, axis
    )


# 2D FOURIER ALGORITHM FUNCTIONS (subgrid to facet)
@dask_wrapper
def prepare_subgrid(subgrid, xM_size, **kwargs):
    """
    Initial shared work per subgrid - no reason to do this per-axis, so always do it for all
    :param subgrid: Subgrid
    :param xM_size: Subgrid size, padded for transfer (internal)
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: the FS term
    """
    padded = pad_mid(pad_mid(subgrid, xM_size, axis=0), xM_size, axis=1)
    ftransformed = fft(fft(padded, axis=0), axis=1)

    return ftransformed


@dask_wrapper
def extract_facet_contribution(FSi, facet_off_j, constants_class, axis, **kwargs):
    """
    Extract contribution of subgrid to a facet

    :param Fsi:
    :param facet_off_j:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param axis: Axis
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: Contribution of facet on the subgrid

    """
    return broadcast(constants_class.Fn, len(FSi.shape), axis) * extract_mid(
        numpy.roll(
            FSi, -facet_off_j * constants_class.xM_size // constants_class.N, axis
        ),
        constants_class.xM_yN_size,
        axis,
    )


@dask_wrapper
def add_subgrid_contribution(
    dims,
    NjSi,
    subgrid_off_i,
    constants_class,
    axis,
    **kwargs,
):
    """
    Add subgrid contribution to a facet

    :param MiNjSi_sum:
    :param NjSi:
    :param subgrid_off:
    :param constants_class: ConstantArrays or DistributedFFT class object containing
                            fundamental and derived parameters
    :param axis:
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return MiNjSi_sum:

    """
    xN_yP_size = constants_class.xMxN_yP_size - constants_class.xM_yP_size
    NjSi_mid = ifft(pad_mid(NjSi, constants_class.xM_yP_size, axis), axis)
    NjSi_temp = pad_mid(NjSi_mid, constants_class.xMxN_yP_size, axis)
    slc1 = create_slice(slice(None), slice(xN_yP_size // 2), dims, axis)
    slc2 = create_slice(slice(None), slice(-xN_yP_size // 2, None), dims, axis)
    NjSi_temp[slc1] = NjSi_mid[slc2]
    NjSi_temp[slc2] = NjSi_mid[slc1]
    NjSi_temp = NjSi_temp * broadcast(
        constants_class.facet_m0_trunc, len(NjSi.shape), axis
    )

    return numpy.roll(
        pad_mid(NjSi_temp, constants_class.yP_size, axis),
        subgrid_off_i * constants_class.yP_size // constants_class.N,
        axis=axis,
    )


@dask_wrapper
def finish_facet(MiNjSi_sum, Fb, facet_B_j, yB_size, axis, **kwargs):
    """
    Obtain finished facet

    :param MiNjSi_sum:
    :param Fb:
    :param facet_B_j:
    :param yB_size: effective facet size
    :param axis:
    :param kwargs: needs to contain the following if dask is used:
            use_dask: True
            nout: <number of function outputs> --> 1

    :return: The finished facet (in BMNAF term)
    """
    return extract_mid(fft(MiNjSi_sum, axis), yB_size, axis) * broadcast(
        Fb * facet_B_j, len(MiNjSi_sum.shape), axis
    )


# COMMON 1D and 2D FUNCTIONS -- SETTING UP FOR ALGORITHM TO RUN
def generate_mask(n_image, ndata_point, true_usable_size, offset):
    """
    Determine the appropriate A/B masks for cutting the subgrid/facet out.
    We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

    :param n_image: total image size in one side (N)
    :param ndata_point: number of data points (nsubgrid or nfacet)
    :param true_usable_size: true usable size (xA_size or yB_size)
    :param offset: subgrid or facet offset (subgrid_off or facet_off)

    :return: mask: subgrid_A or facet_B
    """
    mask = numpy.zeros((ndata_point, true_usable_size), dtype=int)
    subgrid_border = (offset + numpy.hstack([offset[1:], [n_image + offset[0]]])) // 2
    for i in range(ndata_point):
        left = (subgrid_border[i - 1] - offset[i] + true_usable_size // 2) % n_image
        right = subgrid_border[i] - offset[i] + true_usable_size // 2
        assert (
            left >= 0 and right <= true_usable_size
        ), "xA / yB not large enough to cover subgrids / facets!"
        mask[i, left:right] = 1

    return mask
