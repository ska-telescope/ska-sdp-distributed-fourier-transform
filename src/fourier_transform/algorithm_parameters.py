import numpy


class Sizes:
    """
    **target_params contains the following keys:

    :param W: PSWF (prolate-spheroidal wave function) parameter (grid-space support)
    :param fov: field of View
    :param N: total image size
    :param Nx: subgrid spacing: subgrid offsets need to be divisible by Nx
    :param yB_size: effective facet size
    :param yP_size: padded (rough) facet size
    :param xA_size: effective subgrid size
    :param xM_size: padded (rough) subgrid size
    :param yN_size: needed padding

    The class, in addition, derives the following, commonly used sizes:

    xM: TODO ?
    xM_yP_size: (padded subgrid size * padded facet size) / N
    xM_yN_size: (padded subgrid size * padding) / N
    xMxN_yP_size: length of the region to be cut out of the prepared facet data
                  (i.e. len(facet_m0_trunc), where facet_m0_trunc is the mask truncated to a facet (image space))
    xN_yP_size: remainder of the padded facet region after the cut-out region has been subtracted of it
                i.e. xMxN_yP_size - xM_yP_size
    """

    def __init__(self, **target_params):
        # Fundamental sizes and parameters
        self.W = target_params["W"]
        self.fov = target_params["fov"]
        self.N = target_params["N"]
        self.Nx = target_params["Nx"]
        self.yB_size = target_params["yB_size"]
        self.yP_size = target_params["yP_size"]
        self.xA_size = target_params["xA_size"]
        self.xM_size = target_params["xM_size"]
        self.yN_size = target_params["yN_size"]

        self.check_params()

        # Commonly used relative coordinates and derived values
        self.xM = self.xM_size / 2 / self.N

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
            f"xM = {self.xM}\n"
            f"xM_yP_size = {self.xM_yP_size}\n"
            f"xM_yN_size = {self.xM_yN_size}\n"
            f"xMxN_yP_size = {self.xMxN_yP_size}\n"
            f"xN_yP_size = {self.xN_yP_size}"
        )
        return class_string
