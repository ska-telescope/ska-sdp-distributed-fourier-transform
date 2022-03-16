import dask.array
import numpy

from src.fourier_transform.dask_wrapper import dask_wrapper
from src.fourier_transform.fourier_algorithm import extract_mid, fft, ifft, pad_mid, _ith_subgrid_facet_element


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
