"""
TODO: we should probably move the following and the
 information from the related Slack conversation into the docs:

An alternative way of calculating nfacet is::

    if sizes.fov is not None:
        nfacet = int(numpy.ceil(sizes.N * sizes.fov / sizes.yB_size))
        log.info(
            f"{nfacet}x{nfacet} facets for FoV of {sizes.fov} "
            f"({sizes.N * sizes.fov / nfacet / sizes.yB_size * 100}% efficiency)"
        )

This makes sure that if we have a specific FoV we care about, then we don't create facets outside that.
See Slack conversation with Peter:
https://skao.slack.com/archives/C02R9BQFK7W/p1645017383044429
(#proj-sp-2086-dask-distributed-pipeline, Feb 16)
"""
import math
import numpy

from src.fourier_transform.fourier_algorithm import (
    extract_mid,
    coordinates,
    ifft,
    pad_mid,
    anti_aliasing_function,
)
from src.utils import whole


class ConstantParams:
    """
    **fundamental_constants contains the following keys:

    :param W: PSWF (prolate-spheroidal wave function) parameter (grid-space support)
    :param fov: field of view
    :param N: total image size
    :param Nx: subgrid spacing: subgrid offsets need to be divisible by Nx
    :param yB_size: effective facet size
    :param yP_size: padded facet size (pad facet with zeros at margins to reach this size)
    :param xA_size: effective subgrid size
    :param xM_size: padded subgrid size (pad subgrid with zeros at margins to reach this size)
    :param yN_size: padded facet size which evenly divides the image size,
                    used for resampling facets into image space

    The class, in addition, derives the following, commonly used sizes:

    xM_yP_size: (padded subgrid size * padded facet size) / N
    xM_yN_size: (padded subgrid size * padding) / N
    xMxN_yP_size: length of the region to be cut out of the prepared facet data
                  (i.e. len(facet_m0_trunc), where facet_m0_trunc is the mask truncated to a facet (image space))
    xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracted of it
                i.e. xMxN_yP_size - xM_yP_size
    nsubgrid: number of subgrids
    nfacet: number of facets
    """

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

        # Note from Peter: Note that this (xM_yP_size) is xM_yN_size * yP_size / yN_size.
        # So could replace by yN_size, which would be the more fundamental entity.
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


class ConstantArrays(ConstantParams):
    """
    Class that calculates and holds constant arrays.
    See the parent class docstring for description of input parameters.

    It contains the following arrays (in addition to ConstantParams):

        facet_off: facet offset
        subgrid_off: subgrid offset
        facet_B: facet mask
        subgrid_A: subgrid mask
        Fb: Fourier transform of grid correction function
        Fn: Fourier transform of gridding function
        facet_m0_trunc: mask truncated to a facet (image space)
        pswf: prolate spheroidal wave function

    Notes on gridding-related functions (arrays; Fb, Fn, facet_m0_trunc):
        Calculate actual work terms to use. We need both $n$ and $b$ in image space
        In case of gridding: "n": gridding function (except that we have it in image space here)
                             "b": grid correction function.

        Note (Peter W): The reason they're single functions (i.e. we only compute one Fn, Fb and m0
        instead of one per facet/subgrid) is that we assume that they are all the same function,
        just shifted in grid and image space respectively (to the positions of the subgrids and facets)
    """

    def __init__(self, **fundamental_constants):
        super().__init__(**fundamental_constants)

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
        if self._facet_off is None:
            self._facet_off = self.yB_size * numpy.arange(self.nfacet)

        return self._facet_off

    @property
    def subgrid_off(self):
        if self._subgrid_off is None:
            self._subgrid_off = self.xA_size * numpy.arange(self.nsubgrid) + self.Nx

        return self._subgrid_off

    def check_offsets(self):
        assert whole(numpy.outer(self.subgrid_off, self.facet_off) / self.N)
        assert whole(self.facet_off * self.xM_size / self.N)

    def _generate_mask(self, ndata_point, true_usable_size, offset):
        """
        Determine the appropriate A/B masks for cutting the subgrid/facet out.
        We are aiming for full coverage here: Every pixel is part of exactly one subgrid / facet.

        :param ndata_point: number of data points (nsubgrid or nfacet)
        :param true_usable_size: true usable size (xA_size or yB_size)
        :param offset: subgrid or facet offset (subgrid_off or facet_off)

        :return: mask: subgrid_A or facet_B
        """
        mask = numpy.zeros((ndata_point, true_usable_size), dtype=int)
        subgrid_border = (
            offset + numpy.hstack([offset[1:], [self.N + offset[0]]])
        ) // 2
        for i in range(ndata_point):
            left = (subgrid_border[i - 1] - offset[i] + true_usable_size // 2) % self.N
            right = subgrid_border[i] - offset[i] + true_usable_size // 2
            assert (
                left >= 0 and right <= true_usable_size
            ), "xA / yB not large enough to cover subgrids / facets!"
            mask[i, left:right] = 1

        return mask

    @property
    def facet_B(self):
        if self._facet_B is None:
            self._facet_B = self._generate_mask(
                self.nfacet, self.yB_size, self.facet_off
            )

        return self._facet_B

    @property
    def subgrid_A(self):
        if self._subgrid_A is None:
            self._subgrid_A = self._generate_mask(
                self.nsubgrid, self.xA_size, self.subgrid_off
            )

        return self._subgrid_A

    @property
    def Fb(self):
        if self._Fb is None:
            self._Fb = 1 / extract_mid(self.pswf, self.yB_size, axis=0)

        return self._Fb

    @property
    def Fn(self):
        if self._Fn is None:
            self._Fn = self.pswf[
                (self.yN_size // 2)
                % int(self.N / self.xM_size) :: int(self.N / self.xM_size)
            ]

        return self._Fn

    @property
    def facet_m0_trunc(self):
        if self._facet_m0_trunc is None:
            temp_facet_m0_trunc = self.pswf * numpy.sinc(
                coordinates(self.yN_size) * self.xM_size / self.N * self.yN_size
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
    def pswf(self, alpha=0):
        """
        Calculate PSWF (prolate-spheroidal wave function) at the
        full required resolution (facet size)

        :param alpha: TODO: ???, int
        """
        if self._pswf is None:
            self._pswf = anti_aliasing_function(
                self.yN_size, alpha, numpy.pi * self.W / 2
            ).real
            self._pswf /= numpy.prod(
                numpy.arange(2 * alpha - 1, 0, -2, dtype=float)
            )  # double factorial

        return self._pswf


class DistributedFFT(ConstantArrays):
    """
    Placeholder-class that will connect all elements of the distributed FFT code.
    Currently it provides the necessary constants and constant arrays through its parent classes.
    """

    def __init__(self, **fundamental_constants):
        super().__init__(**fundamental_constants)
