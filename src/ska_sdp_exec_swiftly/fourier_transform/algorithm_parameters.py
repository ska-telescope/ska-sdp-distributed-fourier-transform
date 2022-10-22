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

import numpy
import scipy.signal
import scipy.special

from .fourier_algorithm import (
    broadcast,
    coordinates,
    extract_mid,
    fft,
    ifft,
    pad_mid,
)


class SwiftlyCore:
    """
    Streaming Distributed Fourier Transform class

    It takes the fundamental_constants dict as input
    (see BaseParameters class).
    It encompasses all building blocks of the algorithm for
    both subgrid -> facet and facet -> subgrid directions.

    The algorithm was developed for 2D input arrays (images).

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

    def __init__(self, W, N, xM_size, yN_size):
        # Fundamental sizes and parameters
        self.W = W
        self.N = N
        self.xM_size = xM_size
        self.yN_size = yN_size
        self.check_params()

        # Derive subgrid <> facet contribution size
        self.xM_yN_size = self.xM_size * self.yN_size // self.N

        # Calculate constants
        pswf = self._calculate_pswf()
        self._Fb = self._calculate_Fb(pswf)
        self._Fn = self._calculate_Fn(pswf)

    def check_params(self):
        """
        Validate some of the parameters.
        """
        if self.N % self.yN_size != 0:
            raise ValueError(
                f"Image size {self.N} not divisible by "
                f"facet size {self.yN_size}!"
            )
        if self.N % self.xM_size != 0:
            raise ValueError(
                f"Image size {self.N} not divisible by "
                f"subgrid size {self.xM_size}!"
            )
        if (self.xM_size * self.yN_size) % self.N != 0:
            raise ValueError(
                f"Contribution size not integer with "
                f"image size {self.N}, subgrid size {self.xM_size} "
                f"and facet size {self.yN_size}!"
            )

    @property
    def subgrid_off_step(self):
        """
        Returns the base subgrid offset.

        All subgrid offsets must be divisible by this value.
        """
        return self.N // self.yN_size

    @property
    def facet_off_step(self):
        """
        Returns the base facet offset.

        All facet offsets must be divisible by this value.
        """
        return self.N // self.xM_size

    def __str__(self):
        class_string = (
            "Fundamental parameters: \n"
            f"W = {self.W}\n"
            f"N = {self.N}\n"
            f"xM_size = {self.xM_size}\n"
            f"yN_size = {self.yN_size}\n"
            f"\nDerived values: \n"
            f"xM_yN_size = {self.xM_yN_size}\n"
        )
        return class_string

    def _calculate_Fb(self, pswf):
        """
        Calculate the Fourier transform of grid correction function
        """
        return 1 / pswf[1:]

    def _calculate_Fn(self, pswf):
        """
        Calculate the Fourier transform of gridding function
        """
        return pswf[
            (self.yN_size // 2)
            % int(self.N / self.xM_size) :: int(self.N / self.xM_size)
        ]

    def _calculate_pswf(self):
        """
        Calculate 1D PSWF (prolate-spheroidal wave function) at the
        full required resolution (facet size)

        See also: VLA Scientific Memoranda 129, 131, 132
        """
        # alpha: mode parameter (integer) for the PSWF
        # eigenfunctions using zero for zeroth order
        alpha = 0

        # pylint: disable=no-member
        pswf = numpy.empty(self.yN_size, dtype=float)
        coords = 2 * coordinates(self.yN_size)

        # Work around segfault that happens if we ask for large arrays.
        # 500 seems to be (precisely?!) the size of the largest array we
        # can safely fill.
        step = 500
        for i in range(1, self.yN_size, step):
            pswf[i : i + step] = scipy.special.pro_ang1(
                alpha,
                alpha,
                numpy.pi * self.W / 2,
                coords[i : i + step],
            )[0]
        pswf[0] = 0  # zap NaN

        # Normalise (double factorial of alpha)
        pswf /= numpy.prod(numpy.arange(2 * alpha - 1, 0, -2, dtype=float))

        return pswf

    # facet to subgrid algorithm
    def prepare_facet(self, facet, facet_off_elem, axis):
        """
        Calculate the inverse FFT of a padded facet element multiplied by Fb
        (Fb: Fourier transform of grid correction function)

        :param facet: single facet element
        :param Fb: Fourier transform of grid correction function
        :param axis: axis along which operations are performed (0 or 1)
        :param chunk: using chunk mode or not

        :return: TODO: BF? prepared facet
        """

        facet_size = facet.shape[axis]
        BF = pad_mid(
            facet
            * broadcast(
                extract_mid(self._Fb, facet_size, 0), len(facet.shape), axis
            ),
            self.yN_size,
            axis,
        )
        return ifft(numpy.roll(BF, facet_off_elem, axis=axis), axis)

    def extract_facet_contrib_to_subgrid(
        self, BF, subgrid_off_elem, axis
    ):  # pylint: disable=too-many-arguments
        """
        Extract the facet contribution to a subgrid.

        :param BF: TODO: ? prepared facet
        :param subgrid_off_elem: single subgrid offset element
        :param facet_off_elem: single subgrid offset element
        :param axis: axis along which the operations are performed (0 or 1)

        :return: contribution of facet to subgrid
        """

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

    def add_facet_contribution(self, facet_contrib, facet_off_elem, axis):
        """
        Further transforms facet contributions, which then will be summed up.

        :param facet_contrib: array-chunk of individual facet contributions
        :param facet_off_elem: facet offset for the facet_contrib array chunk
        :param axis: axis along which the operations are performed (0 or 1)

        :return: TODO??
        """

        scaled_facet_off_elem = facet_off_elem * self.xM_size // self.N

        FNMBF = broadcast(
            self._Fn, len(facet_contrib.shape), axis
        ) * numpy.roll(
            fft(facet_contrib, axis), -scaled_facet_off_elem, axis=axis
        )

        return numpy.roll(
            pad_mid(FNMBF, self.xM_size, axis),
            scaled_facet_off_elem,
            axis=axis,
        )

    def finish_subgrid(
        self, summed_facets, subgrid_off_elem, subgrid_size, subgrid_masks=None
    ):
        """
        Obtain finished subgrid. Operation performed for all axes.

        :param summed_facets: summed facets contributing to this subgrid
        :param subgrid_off: subgrid offset per axis
        :param subgrid_size: subgrid size
        :param subgrid_masks: subgrid mask per axis (optional)

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
                subgrid_size,
                axis=axis,
            )

            # Apply subgrid mask if requested
            if subgrid_masks is not None:
                tmp *= broadcast(subgrid_masks[axis], dims, axis)

        return tmp

    # subgrid to facet algorithm
    def prepare_subgrid(self, subgrid, subgrid_off_elem):
        """
        Calculate the FFT of a padded subgrid element.
        No reason to do this per-axis, so always do it for all axes.

        :param subgrid: single subgrid array element
        :param subgrid_off_elem: subgrid offsets (tuple)

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

    def extract_subgrid_contrib_to_facet(self, FSi, facet_off_elem, axis):
        """
        Extract contribution of subgrid to a facet.

        :param Fsi: Prepared subgrid in image space (see prepare_facet)
        :param facet_off_elem: facet offset
        :param axis: axis along which the operations are performed (0 or 1)

        :return: Contribution of subgrid to facet

        """

        # Align with image zero in image space
        FSi = numpy.roll(
            FSi,
            -facet_off_elem * self.xM_size // self.N,
            axis,
        )

        FNjSi = broadcast(self._Fn, len(FSi.shape), axis) * extract_mid(
            FSi,
            self.xM_yN_size,
            axis,
        )

        return numpy.roll(
            FNjSi, facet_off_elem * self.xM_size // self.N, axis=axis
        )

    # pylint: disable=too-many-arguments
    def add_subgrid_contribution(
        self,
        subgrid_contrib,
        subgrid_off_elem,
        axis,
    ):
        """
        Sum up subgrid contributions to a facet.

        :param subgrid_contrib: Subgrid contribution to this facet (see
                extract_subgrid_contrib_to_facet)
        :param subgrid_off_elem: subgrid offset
        :param facet_m0_trunc: mask truncated to a facet (image space)
        :param axis: axis along which operations are performed (0 or 1)

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

    def finish_facet(
        self, MiNjSi_sum, facet_off_elem, facet_size, facet_B_mask_elem, axis
    ):
        """
        Obtain finished facet.

        It extracts from the padded facet (obtained from subgrid via FFT)
        the true-sized facet and multiplies with masked Fb.

        :param MiNjSi_sum: sum of subgrid contributions to a facet
        :param facet_size: facet size
        :param facet_off_elem: facet offset
        :param facet_B_mask_elem: a facet mask element
        :param axis: axis along which operations are performed (0 or 1)

        :return: finished (approximate) facet element
        """

        if facet_B_mask_elem is not None:
            facet_mask = broadcast(
                extract_mid(self._Fb, facet_size, 0) * facet_B_mask_elem,
                len(MiNjSi_sum.shape),
                axis,
            )
        else:
            facet_mask = broadcast(
                extract_mid(self._Fb, facet_size, 0),
                len(MiNjSi_sum.shape),
                axis,
            )

        return facet_mask * extract_mid(
            numpy.roll(fft(MiNjSi_sum, axis), -facet_off_elem, axis=axis),
            facet_size,
            axis,
        )
