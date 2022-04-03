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

# pure function; broadcast,; create_slice,; fft,
from src.fourier_transform.fourier_algorithm import (
    add_facet_contribution,
    add_subgrid_contribution,
    coordinates,
    extract_facet_contrib_to_subgrid,
    extract_mid,
    extract_subgrid_contrib_to_facet,
    finish_facet,
    finish_subgrid,
    ifft,
    pad_mid,
    prepare_facet,
    prepare_subgrid,
)


class BaseParameters:
    """
    **fundamental_constants contains the following keys:

    :param W: PSWF (prolate-spheroidal wave function)
              parameter (grid-space support)
    :param fov: field of view
    :param N: total image size
    :param Nx: subgrid spacing: subgrid offsets need to be divisible by Nx
    :param yB_size: effective facet size
    :param yP_size: padded facet size (pad facet with zeros
                    at margins to reach this size)
    :param xA_size: effective subgrid size
    :param xM_size: padded subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param yN_size: padded facet size which evenly divides the image size,
                    used for resampling facets into image space

    A / x --> grid (frequency) space; B / y --> image (facet) space

    The class, in addition, derives the following, commonly used sizes:

    :param xM_yP_size: (padded subgrid size * padded facet size) / N
    :param xM_yN_size: (padded subgrid size * padding) / N
    :param xMxN_yP_size: length of the region to be cut out of the
                         prepared facet data (i.e. len(facet_m0_trunc),
                         where facet_m0_trunc is the mask truncated
                         to a facet (image space))
    :param xN_yP_size: remainder of the padded facet region after the cut-out
                       region has been subtracted of it
                       i.e. xMxN_yP_size - xM_yP_size
    :param nsubgrid: number of subgrids
    :param nfacet: number of facets
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, **fundamental_constants):
        # Fundamental sizes and parameters
        self.W = fundamental_constants["W"]
        self.fov = fundamental_constants["fov"]
        self.N = fundamental_constants["N"]
        self.Nx = fundamental_constants["Nx"]
        self.yB_size = fundamental_constants["yB_size"]
        self.yP_size = fundamental_constants["yP_size"]
        self.xA_size = fundamental_constants["xA_size"]
        self.xM_size = fundamental_constants["xM_size"]
        self.yN_size = fundamental_constants["yN_size"]

        self.check_params()

        # Commonly used relative coordinates and derived values

        # Note from Peter: Note that this (xM_yP_size) is
        #   xM_yN_size * yP_size / yN_size.
        # So could replace by yN_size, which would be the
        # more fundamental entity.
        # TODO ^
        self.xM_yP_size = self.xM_size * self.yP_size // self.N
        # same note as above xM_yP_size; could be replaced with xM_size
        self.xM_yN_size = self.xM_size * self.yN_size // self.N

        xN_size = self.N * self.W / self.yN_size
        self.xMxN_yP_size = self.xM_yP_size + int(
            2 * numpy.ceil(xN_size * self.yP_size / self.N / 2)
        )

        self.xN_yP_size = self.xMxN_yP_size - self.xM_yP_size

        self.nsubgrid = int(math.ceil(self.N / self.xA_size))
        self.nfacet = int(math.ceil(self.N / self.yB_size))

    def check_params(self):
        """
        Validate some of the parameters.
        """
        if not (self.xM_size * self.yN_size) % self.N == 0:
            raise ValueError

    def __str__(self):
        class_string = (
            "Fundamental parameters: \n"
            f"W = {self.W}\n"
            f"fov = {self.fov}\n"
            f"N = {self.N}\n"
            f"Nx = {self.Nx}\n"
            f"yB_size = {self.yB_size}\n"
            f"yP_size = {self.yP_size}\n"
            f"xA_size = {self.xA_size}\n"
            f"xM_size = {self.xM_size}\n"
            f"yN_size = {self.yN_size}\n"
            f"\nDerived values: \n"
            f"xM_yP_size = {self.xM_yP_size}\n"
            f"xM_yN_size = {self.xM_yN_size}\n"
            f"xMxN_yP_size = {self.xMxN_yP_size}\n"
            f"xN_yP_size = {self.xN_yP_size}\n"
            f"nsubgrid = {self.nsubgrid}\n"
            f"nfacet = {self.nfacet}"
        )
        return class_string


class BaseArrays(BaseParameters):
    """
    Class that calculates and holds fundamental constant arrays.
    See the parent class docstring for description of input parameters.

    It contains the following arrays (in addition to BaseParameters):

    :param facet_off: facet offset
    :param subgrid_off: subgrid offset
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
    __slots__ = ()

    def __init__(self, **fundamental_constants):
        super(BaseArrays, self).__init__(**fundamental_constants)

        self._facet_off = None
        self._subgrid_off = None
        self._facet_B = None
        self._subgrid_A = None
        self._Fb = None
        self._Fn = None
        self._facet_m0_trunc = None
        self._pswf = None

    @property
    def facet_off(self):
        """
        Facet offset array
        """
        if self._facet_off is None:
            self._facet_off = self.yB_size * numpy.arange(self.nfacet)

        return self._facet_off

    @property
    def subgrid_off(self):
        """
        Subgrid offset array
        """
        if self._subgrid_off is None:
            self._subgrid_off = (
                self.xA_size * numpy.arange(self.nsubgrid) + self.Nx
            )

        return self._subgrid_off

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

    @property
    def facet_B(self):
        """
        Facet mask
        """
        if self._facet_B is None:
            self._facet_B = self._generate_mask(self.yB_size, self.facet_off)

        return self._facet_B

    @property
    def subgrid_A(self):
        """
        Subgrid mask
        """
        if self._subgrid_A is None:
            self._subgrid_A = self._generate_mask(
                self.xA_size, self.subgrid_off
            )

        return self._subgrid_A

    @property
    def Fb(self):
        """
        Fourier transform of grid correction function
        """
        if self._Fb is None:
            self._Fb = 1 / extract_mid(self.pswf, self.yB_size, axis=0)

        return self._Fb

    @property
    def Fn(self):
        """
        Fourier transform of gridding function
        """
        if self._Fn is None:
            self._Fn = self.pswf[
                (self.yN_size // 2)
                % int(self.N / self.xM_size) :: int(self.N / self.xM_size)
            ]

        return self._Fn

    @property
    def facet_m0_trunc(self):
        """
        Mask truncated to a facet (image space)
        """
        if self._facet_m0_trunc is None:
            temp_facet_m0_trunc = self.pswf * numpy.sinc(
                coordinates(self.yN_size)
                * self.xM_size
                / self.N
                * self.yN_size
            )
            self._facet_m0_trunc = (
                self.xM_size
                * self.yP_size
                / self.N
                * extract_mid(
                    ifft(
                        pad_mid(temp_facet_m0_trunc, self.yP_size, axis=0),
                        axis=0,
                    ),
                    self.xMxN_yP_size,
                    axis=0,
                ).real
            )

        return self._facet_m0_trunc

    @property
    def pswf(self):
        """
        Calculate 1D PSWF (prolate-spheroidal wave function) at the
        full required resolution (facet size)

        See also: VLA Scientific Memoranda 129, 131, 132
        """
        # alpha: mode parameter (integer) for the PSWF
        # eigenfunctions using zero for zeroth order
        alpha = 0

        if self._pswf is None:
            # pylint: disable=no-member
            pswf = scipy.special.pro_ang1(
                alpha,
                alpha,
                numpy.pi * self.W / 2,
                2 * coordinates(self.yN_size),
            )[0]
            pswf[0] = 0  # zap NaN

            self._pswf = pswf.real
            self._pswf /= numpy.prod(
                numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
            )  # double factorial

        return self._pswf


class SparseFourierTransform(BaseArrays):
    """
    Sparse Fourier Transform class

    "Sparse" because instead of individual points in frequency
    space we handle entire sub-grids, when running FT.

    It takes the fundamental_constants dict as input
    (see BaseParameters class).
    It encompasses all building blocks of the algorithm for
    both subgrid -> facet and facet -> subgrid directions.

    The algorithm was developed for 2D input arrays (images).
    """

    __slots__ = ()

    def __init__(self, **fundamental_constants):
        super(SparseFourierTransform, self).__init__(**fundamental_constants)

    # facet to subgrid algorithm
    # @dask_wrapper
    def prepare_facet(self, facet, axis, **kwargs):
        """
        Calculate the inverse FFT of a padded facet element multiplied by Fb
        (Fb: Fourier transform of grid correction function)

        :param facet: single facet element
        :param axis: axis along which operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: TODO: BF? prepared facet
        """
        res = dask_wrapper(prepare_facet)(
            facet, axis, self.Fb, self.yP_size, **kwargs
        )
        return res

    def extract_facet_contrib_to_subgrid(
        self,
        BF,
        axis,
        subgrid_off_elem,
        **kwargs,
    ):
        """
        Extract the facet contribution to a subgrid.

        :param BF: TODO: ? prepared facet
        :param axis: axis along which the operations are performed (0 or 1)
        :param subgrid_off_elem: single subgrid offset element
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: contribution of facet to subgrid
        """
        res = dask_wrapper(extract_facet_contrib_to_subgrid)(
            BF,
            axis,
            subgrid_off_elem,
            self.yP_size,
            self.xMxN_yP_size,
            self.xM_yP_size,
            self.xM_yN_size,
            self.N,
            self.Fn,
            self.facet_m0_trunc,
            **kwargs,
        )

        return res

    def add_facet_contribution(
        self, facet_contrib, facet_off_elem, axis, **kwargs
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
        res = dask_wrapper(add_facet_contribution)(
            facet_contrib, facet_off_elem, axis, self.xM_size, self.N, **kwargs
        )
        return res

    def finish_subgrid(
        self, summed_facets, subgrid_mask1, subgrid_mask2, **kwargs
    ):
        """
        Obtain finished subgrid.
        Operation performed for both axis
        (only works on 2D arrays in its current form).

        :param summed_facets: summed facets contributing to thins subgrid
        :param subgrid_mask1: ith subgrid mask element
        :param subgrid_mask2: (i+1)th subgrid mask element
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: approximate subgrid element
        """
        res = dask_wrapper(finish_subgrid)(
            summed_facets, subgrid_mask1, subgrid_mask2, self.xA_size, **kwargs
        )
        return res

    # subgrid to facet algorithm
    def prepare_subgrid(self, subgrid, **kwargs):
        """
        Calculate the FFT of a padded subgrid element.
        No reason to do this per-axis, so always do it for both axis.
        (Note: it will only work for 2D subgrid arrays)

        :param subgrid: single subgrid array element
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: TODO: the FS ??? term
        """
        res = dask_wrapper(prepare_subgrid)(subgrid, self.xM_size, **kwargs)

        return res

    def extract_subgrid_contrib_to_facet(
        self, FSi, facet_off_elem, axis, **kwargs
    ):
        """
        Extract contribution of subgrid to a facet.

        :param Fsi: TODO???
        :param facet_off_elem: single facet offset element
        :param axis: axis along which the operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: Contribution of subgrid to facet

        """
        res = dask_wrapper(extract_subgrid_contrib_to_facet)(
            FSi,
            facet_off_elem,
            axis,
            self.xM_size,
            self.xM_yN_size,
            self.N,
            self.Fn,
            **kwargs,
        )
        return res

    def add_subgrid_contribution(
        self,
        dims,
        NjSi,
        subgrid_off_elem,
        axis,
        **kwargs,
    ):
        """
        Further transform subgrid contributions, which are then summed up.

        :param dims: length of tuple to be produced by create_slice
                     (i.e. number of dimensions); int
        :param NjSi: TODO
        :param subgrid_off_elem: single subgrid offset element
        :param axis: axis along which operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return summed subgrid contributions

        """
        res = dask_wrapper(add_subgrid_contribution)(
            dims,
            NjSi,
            subgrid_off_elem,
            axis,
            self.xMxN_yP_size,
            self.xM_yP_size,
            self.yP_size,
            self.N,
            self.facet_m0_trunc,
            **kwargs,
        )

        return res

    def finish_facet(self, MiNjSi_sum, facet_B_elem, axis, **kwargs):
        """
        Obtain finished facet.

        It extracts from the padded facet (obtained from subgrid via FFT)
        the true-sized facet and multiplies with masked Fb.
        (Fb: Fourier transform of grid correction function)

        :param MiNjSi_sum: sum of subgrid contributions to a facet
        :param facet_B_elem: a facet mask element
        :param axis: axis along which operations are performed (0 or 1)
        :param kwargs: needs to contain the following if dask is used:
                use_dask: True
                nout: <number of function outputs> --> 1

        :return: finished (approximate) facet element
        """
        res = dask_wrapper(finish_facet)(
            MiNjSi_sum, facet_B_elem, axis, self.yB_size, self.Fb, **kwargs
        )
        return res
