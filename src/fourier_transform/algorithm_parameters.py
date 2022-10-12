# pylint: disable=unused-argument
"""
The main data classes are listed in this module.

Note: An alternative way of calculating nfacet is::

    if sizes.fov is not None:
        nfacet = int(numpy.ceil(sizes.N * sizes.fov / sizes.yB_size))
        log.info(
            f"{nfacet}x{nfacet} facets for FoV of {sizes.fov} "
            f"({sizes.N * sizes.fov / nfacet / sizes.yB_size * 100}% efficiency)"
        )

This makes sure that if we have a specific FoV we care about,
then we don't create facets outside that.
"""  # noqa: E501
import math

import numpy
import scipy.signal
import scipy.special

from src.fourier_transform.dask_wrapper import dask_wrapper
from src.fourier_transform.fourier_algorithm import (
    broadcast,
    coordinates,
    create_slice,
    extract_mid,
    fft,
    ifft,
    pad_mid,
    roll_and_extract_mid_axis,
)


class BaseParameters:
    """
    **fundamental_constants contains the following keys:

    :param W: PSWF (prolate-spheroidal wave function)
              parameter (grid-space support)
    :param fov: field of view
    :param N: total image size
    :param yB_size: effective facet size
    :param xA_size: effective subgrid size
    :param xM_size: padded subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param yN_size: padded facet size which evenly divides the image size,
                    used for resampling facets into image space

    A / x --> grid (frequency) space; B / y --> image (facet) space

    The class, in addition, derives the following,
    commonly used sizes (integers), and offset arrays:

    :param xM_yN_size: (padded subgrid size * padding) / N
    :param nsubgrid: number of subgrids
    :param nfacet: number of facets
    :param facet_off: facet offset (numpy array)
    :param subgrid_off: subgrid offset (numpy array)
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, **fundamental_constants):
        # Fundamental sizes and parameters
        self.W = fundamental_constants["W"]
        self.fov = fundamental_constants["fov"]
        self.N = fundamental_constants["N"]
        self.yB_size = fundamental_constants["yB_size"]
        self.xA_size = fundamental_constants["xA_size"]
        self.xM_size = fundamental_constants["xM_size"]
        self.yN_size = fundamental_constants["yN_size"]

        self.check_params()

        # Derive subgrid <> facet contribution size
        self.xM_yN_size = self.xM_size * self.yN_size // self.N

        # Subgrid counts and offsets assuming complete re-distribution
        self.nsubgrid = int(math.ceil(self.N / self.xA_size))
        self.nfacet = int(math.ceil(self.N / self.yB_size))

        self.facet_off = self.calculate_facet_off()
        self.subgrid_off = self.calculate_subgrid_off()

    def check_params(self):
        """
        Validate some of the parameters.
        """
        if not self.N % self.yN_size == 0:
            raise ValueError
        if not self.N % self.xM_size == 0:
            raise ValueError

    def calculate_facet_off(self):
        """
        Calculate facet offset array
        """
        facet_off = self.yB_size * numpy.arange(self.nfacet)
        assert numpy.all(facet_off % (self.N // self.xM_size) == 0)
        return facet_off

    def calculate_subgrid_off(self):
        """
        Calculate subgrid offset array
        """
        subgrid_off = self.xA_size * numpy.arange(self.nsubgrid)
        assert numpy.all(subgrid_off % (self.N // self.yN_size) == 0)
        return subgrid_off

    def __str__(self):
        class_string = (
            "Fundamental parameters: \n"
            f"W = {self.W}\n"
            f"fov = {self.fov}\n"
            f"N = {self.N}\n"
            f"yB_size = {self.yB_size}\n"
            f"xA_size = {self.xA_size}\n"
            f"xM_size = {self.xM_size}\n"
            f"yN_size = {self.yN_size}\n"
            f"\nDerived values: \n"
            f"xM_yN_size = {self.xM_yN_size}\n"
            f"nsubgrid = {self.nsubgrid}\n"
            f"nfacet = {self.nfacet}\n"
            f"facet_off = {self.facet_off}\n"
            f"subgrid_off = {self.subgrid_off}"
        )
        return class_string


class BaseArrays(BaseParameters):
    """
    Class that calculates and holds fundamental constant arrays.
    See the parent class docstring for description of input parameters.

    It contains the following arrays (in addition to BaseParameters):

    :param facet_B: facet mask
    :param subgrid_A: subgrid mask
    :param Fb: Fourier transform of grid correction function
    :param Fn: Fourier transform of gridding function
    :param facet_m0_trunc: mask truncated to a facet (image space)
    :param pswf: prolate spheroidal wave function

    Notes on gridding-related functions (arrays; Fb, Fn, facet_m0_trunc):

        Calculate actual work terms to use.
        We need both $n$ and $b$ in image space
        In case of gridding: "n": gridding function (in image space)
                             "b": grid correction function.

        Note (Peter W): The reason they're single functions
        (i.e. we only compute one Fn, Fb and m0 instead of one
        per facet/subgrid) is that we assume that they are all
        the same function, just shifted in grid and image space
        respectively (to the positions of the subgrids and facets)
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, **fundamental_constants):
        super().__init__(**fundamental_constants)

        self.facet_B = self.calculate_facet_B()
        self.subgrid_A = self.calculate_subgrid_A()
        self.pswf = self.calculate_pswf()
        self.Fb = self.calculate_Fb()
        self.Fn = self.calculate_Fn()

    def _generate_mask(self, mask_size, offsets):
        """
        Determine the appropriate masks for cutting out subgrids/facets.
        For each offset in offsets, a mask is generated of size mask_size.
        The mask is centred around the specific offset.

        :param mask_size: size of the required mask (xA_size or yB_size)
        :param offsets: array of subgrid or facet offsets
                        (subgrid_off or facet_off)

        :return: mask (subgrid_A or facet_B)
        """
        mask = numpy.zeros((len(offsets), mask_size), dtype=int)
        border = (
            offsets + numpy.hstack([offsets[1:], [self.N + offsets[0]]])
        ) // 2
        for i, offset in enumerate(offsets):
            left = (border[i - 1] - offset + mask_size // 2) % self.N
            right = border[i] - offset + mask_size // 2

            if not left >= 0 and right <= mask_size:
                raise ValueError(
                    "Mask size not large enough to cover subgrids / facets!"
                )

            mask[i, left:right] = 1

        return mask

    def calculate_facet_B(self):
        """
        Calculate facet mask
        """
        facet_B = self._generate_mask(self.yB_size, self.facet_off)
        return facet_B

    def calculate_subgrid_A(self):
        """
        Calculate subgrid mask
        """
        subgrid_A = self._generate_mask(self.xA_size, self.subgrid_off)
        return subgrid_A

    def calculate_Fb(self):
        """
        Calculate the Fourier transform of grid correction function
        """
        Fb = 1 / extract_mid(self.pswf, self.yB_size, axis=0)
        return Fb

    def calculate_Fn(self):
        """
        Calculate the Fourier transform of gridding function
        """
        Fn = self.pswf[
            (self.yN_size // 2)
            % int(self.N / self.xM_size) :: int(self.N / self.xM_size)
        ]
        return Fn

    def calculate_pswf(self):
        """
        Calculate 1D PSWF (prolate-spheroidal wave function) at the
        full required resolution (facet size)

        See also: VLA Scientific Memoranda 129, 131, 132
        """
        # alpha: mode parameter (integer) for the PSWF
        # eigenfunctions using zero for zeroth order
        alpha = 0

        # pylint: disable=no-member
        pswf = scipy.special.pro_ang1(
            alpha,
            alpha,
            numpy.pi * self.W / 2,
            2 * coordinates(self.yN_size),
        )[0]
        pswf[0] = 0  # zap NaN

        pswf = pswf.real
        pswf /= numpy.prod(
            numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
        )  # double factorial

        return pswf


class StreamingDistributedFFT(BaseParameters):
    """
    Streaming Distributed Fourier Transform class

    It takes the fundamental_constants dict as input
    (see BaseParameters class).
    It encompasses all building blocks of the algorithm for
    both subgrid -> facet and facet -> subgrid directions.

    The algorithm was developed for 2D input arrays (images).
    """

    # facet to subgrid algorithm
    @dask_wrapper
    def prepare_facet(self, facet, facet_off_elem, Fb, axis, **kwargs):
        """
        Calculate the inverse FFT of a padded facet element multiplied by Fb
        (Fb: Fourier transform of grid correction function)

        :param facet: single facet element
        :param Fb: Fourier transform of grid correction function
        :param axis: axis along which operations are performed (0 or 1)
        :param chunk: using chunk mode or not
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: TODO: BF? prepared facet
        """

        BF = pad_mid(
            facet * broadcast(Fb, len(facet.shape), axis),
            self.yN_size,
            axis,
        )
        return ifft(numpy.roll(BF, facet_off_elem, axis=axis), axis)

    @dask_wrapper
    def extract_facet_contrib_to_subgrid(
        self,
        BF,
        subgrid_off_elem,
        axis,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        """
        Extract the facet contribution to a subgrid.

        :param BF: TODO: ? prepared facet
        :param subgrid_off_elem: single subgrid offset element
        :param facet_off_elem: single subgrid offset element
        :param Fn: Fourier transform of gridding function
        :param axis: axis along which the operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: contribution of facet to subgrid
        """
        dims = len(BF.shape)

        scaled_subgrid_off_elem = subgrid_off_elem * self.yN_size // self.N
        return numpy.roll(
            extract_mid(
                numpy.roll(BF, -scaled_subgrid_off_elem, axis=axis),
                self.xM_yN_size,
                axis,
            ),
            scaled_subgrid_off_elem,
            axis,
        )

    @dask_wrapper
    def add_facet_contribution(
        self, facet_contrib, facet_off_elem, Fn, axis, **kwargs
    ):
        """
        Further transforms facet contributions, which then will be summed up.

        :param facet_contrib: array-chunk of individual facet contributions
        :param facet_off_elem: facet offset for the facet_contrib array chunk
        :param axis: axis along which the operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: TODO??
        """

        scaled_facet_off_elem = facet_off_elem * self.xM_size // self.N

        FNMBF = broadcast(Fn, len(facet_contrib.shape), axis) * numpy.roll(
            fft(facet_contrib, axis), -scaled_facet_off_elem, axis=axis
        )

        return numpy.roll(
            pad_mid(FNMBF, self.xM_size, axis),
            scaled_facet_off_elem,
            axis=axis,
        )

    @dask_wrapper
    def finish_subgrid(
        self,
        summed_facets,
        subgrid_off_elem,
        subgrid_masks=None,
        **kwargs,
    ):
        """
        Obtain finished subgrid. Operation performed for all axes.

        :param summed_facets: summed facets contributing to thins subgrid
        :param subgrid_masks: subgrid mask per axis (optional)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: approximate subgrid element
        """

        tmp = summed_facets
        dims = len(summed_facets.shape)

        if not isinstance(subgrid_off_elem, list):
            if dims != 1:
                raise ValueError(
                    "Subgrid offset must be given for every dimension!"
                )
            subgrid_off_elem = [subgrid_off_elem]

        # Loop operation over all axes
        for axis in range(dims):
            tmp = extract_mid(
                numpy.roll(
                    ifft(tmp, axis=axis), -subgrid_off_elem[axis], axis=axis
                ),
                self.xA_size,
                axis=axis,
            )

            # Apply subgrid mask if requested
            if subgrid_masks is not None:
                tmp *= broadcast(subgrid_masks[axis], dims, axis)

        return tmp

    # subgrid to facet algorithm
    @dask_wrapper
    def prepare_subgrid(self, subgrid, subgrid_off_elem, **kwargs):
        """
        Calculate the FFT of a padded subgrid element.
        No reason to do this per-axis, so always do it for all axes.

        :param subgrid: single subgrid array element
        :param subgrid_off_elem: subgrid offsets (tuple)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: Padded subgrid in image space
        """

        tmp = subgrid
        dims = len(subgrid.shape)
        if dims == 1 and not isinstance(subgrid_off_elem, tuple):
            subgrid_off_elem = (subgrid_off_elem,)
        if len(subgrid_off_elem) != dims:
            raise ValueError(
                "Dimensionality mismatch between subgrid and offsets!"
            )

        # Loop operation over all axes
        for axis in range(dims):

            # Pad & align with global zero modulo xM_size
            tmp = numpy.roll(
                pad_mid(tmp, self.xM_size, axis=axis),
                subgrid_off_elem[axis],
                axis=axis,
            )

            # Bring into image space
            tmp = fft(tmp, axis=axis)

        return tmp

    @dask_wrapper
    def extract_subgrid_contrib_to_facet(
        self, FSi, facet_off_elem, Fn, axis, **kwargs
    ):
        """
        Extract contribution of subgrid to a facet.

        :param Fsi: Prepared subgrid in image space (see prepare_facet)
        :param facet_off_elem: facet offset
        :param Fn: window function in image space
        :param axis: axis along which the operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: Contribution of subgrid to facet

        """

        # Align with image zero in image space
        FSi = numpy.roll(
            FSi,
            -facet_off_elem * self.xM_size // self.N,
            axis,
        )

        FNjSi = broadcast(Fn, len(FSi.shape), axis) * extract_mid(
            FSi,
            self.xM_yN_size,
            axis,
        )

        return numpy.roll(
            FNjSi, facet_off_elem * self.xM_size // self.N, axis=axis
        )

    # pylint: disable=too-many-arguments
    @dask_wrapper
    def add_subgrid_contribution(
        self,
        subgrid_contrib,
        subgrid_off_elem,
        axis,
        **kwargs,
    ):
        """
        Sum up subgrid contributions to a facet.

        :param subgrid_contrib: Subgrid contribution to this facet (see
                extract_subgrid_contrib_to_facet)
        :param subgrid_off_elem: subgrid offset
        :param facet_m0_trunc: mask truncated to a facet (image space)
        :param axis: axis along which operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return summed subgrid contributions

        """

        # Bring subgrid contribution into frequency space
        NjSi_temp = ifft(subgrid_contrib, axis)
        MiNjSi = numpy.roll(
            NjSi_temp,
            -subgrid_off_elem * self.yN_size // self.N,
            axis=axis,
        )

        # Finally put at the right place in the (full) frequency space
        # at padded facet resolution
        return numpy.roll(
            pad_mid(
                MiNjSi,
                self.yN_size,
                axis,
            ),
            subgrid_off_elem * self.yN_size // self.N,
            axis=axis,
        )

    @dask_wrapper
    def finish_facet(
        self, MiNjSi_sum, facet_off_elem, facet_B_mask_elem, Fb, axis, **kwargs
    ):
        """
        Obtain finished facet.

        It extracts from the padded facet (obtained from subgrid via FFT)
        the true-sized facet and multiplies with masked Fb.
        (Fb: Fourier transform of grid correction function)

        :param MiNjSi_sum: sum of subgrid contributions to a facet
        :param facet_off_elem: facet offset
        :param facet_B_mask_elem: a facet mask element
        :param Fb: Fourier transform of grid correction function
        :param axis: axis along which operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: finished (approximate) facet element
        """
        return extract_mid(
            numpy.roll(fft(MiNjSi_sum, axis), -facet_off_elem, axis=axis),
            self.yB_size,
            axis,
        ) * broadcast(
            Fb * facet_B_mask_elem,
            len(MiNjSi_sum.shape),
            axis,
        )
