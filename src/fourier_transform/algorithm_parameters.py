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


class Sizes:
    """
    **fundamental_constants contains the following keys:

    :param W: PSWF (prolate-spheroidal wave function) parameter (grid-space support)
    :param fov: field of view
    :param N: total image size
    :param Nx: subgrid spacing: subgrid offsets need to be divisible by Nx
    :param yB_size: effective facet size
    :param yP_size: padded (rough) facet size
    :param xA_size: effective subgrid size
    :param xM_size: padded (rough) subgrid size
    :param yN_size: needed padding

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
