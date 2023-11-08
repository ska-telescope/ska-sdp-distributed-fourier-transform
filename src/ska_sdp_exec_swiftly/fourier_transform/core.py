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

    It encompasses all processing functions for both the subgrid -> facet and
    facet -> subgrid directions, with support for 1D and 2D arrays.

    :param W: PSWF (prolate-spheroidal wave function)
              parameter (grid-space support)
    :param fov: field of view
    :param N: total image size
    :param xM_size: padded subgrid size (pad subgrid with zeros
                    at margins to reach this size)
    :param yN_size: padded facet size which evenly divides the image size,
                    used for resampling facets into image space
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
    def prepare_facet(self, facet, facet_off, axis):
        """
        Prepare facet for extracting subgrid contribution

        This is a relatively expensive operation, both in terms of computation
        and generated data. It should therefore where possible be used for
        multiple :py:func:`extract_facet_contrib_to_subgrid` calls.

        :param facet: single facet element
        :param subgrid_off: subgrid offset
        :param axis: axis along which operations are performed (0 or 1)
        :return: prepared facet data
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
        return ifft(numpy.roll(BF, facet_off, axis=axis), axis)

    def extract_facet_contrib_to_subgrid(
        self, prep_facet, subgrid_off, axis
    ):  # pylint: disable=too-many-arguments
        """
        Extract the facet contribution to a subgrid.

        :param prep_facet: prepared facet (see :py:func:`prepare_facet`)
        :param subgrid_off: subgrid offset
        :param axis: axis along which the operations are performed (0 or 1)

        :return: compact representation of contribution of facet to subgrid
        """

        scaled_subgrid_off = subgrid_off * self.yN_size // self.N
        return numpy.roll(
            extract_mid(
                numpy.roll(prep_facet, -scaled_subgrid_off, axis=axis),
                self.xM_yN_size,
                axis,
            ),
            scaled_subgrid_off,
            axis,
        )

    def add_facet_contribution(self, facet_contrib, facet_off, axis):
        """
        Further transforms facet contributions, which then will be summed up.

        :param facet_contrib: array-chunk of individual facet contributions
        :param facet_off: facet offset for the facet_contrib array chunk
        :param axis: axis along which the operations are performed (0 or 1)
        :return: addition to subgrid
        """

        scaled_facet_off = facet_off * self.xM_size // self.N

        FNMBF = broadcast(
            self._Fn, len(facet_contrib.shape), axis
        ) * numpy.roll(fft(facet_contrib, axis), -scaled_facet_off, axis=axis)

        return numpy.roll(
            pad_mid(FNMBF, self.xM_size, axis),
            scaled_facet_off,
            axis=axis,
        )

    def finish_subgrid(
        self, summed_contribs, subgrid_off, subgrid_size, out=None
    ):
        """
        Obtain finished subgrid. Operation performed for all axes.

        :param summed_contribs: summed facet contributions to this subgrid
        :param subgrid_off: subgrid offset per axis
        :param subgrid_size: subgrid size

        :return: approximate subgrid element
        """

        dims = len(summed_contribs.shape)
        if not isinstance(subgrid_off, list):
            if dims != 1:
                raise ValueError(
                    "Subgrid offset must be given for every dimension!"
                )
            subgrid_off = [subgrid_off]

        # Loop operation over all axes
        tmp = summed_contribs
        for axis in range(dims):
            tmp = extract_mid(
                numpy.roll(
                    ifft(tmp, axis=axis), -subgrid_off[axis], axis=axis
                ),
                subgrid_size,
                axis=axis,
            )

        return tmp

    # subgrid to facet algorithm
    def prepare_subgrid(self, subgrid, subgrid_off):
        """
        Calculate the FFT of a padded subgrid element.
        No reason to do this per-axis, so always do it for all axes.

        :param subgrid: single subgrid array element
        :param subgrid_off: subgrid offsets (tuple)

        :return: Padded subgrid in image space
        """

        tmp = subgrid
        dims = len(subgrid.shape)
        if dims == 1 and not isinstance(subgrid_off, tuple):
            subgrid_off = (subgrid_off,)
        if len(subgrid_off) != dims:
            raise ValueError(
                "Dimensionality mismatch between subgrid and offsets!"
            )

        # Loop operation over all axes
        for axis in range(dims):
            # Pad & align with global zero modulo xM_size
            tmp = numpy.roll(
                pad_mid(tmp, self.xM_size, axis=axis),
                subgrid_off[axis],
                axis=axis,
            )

            # Bring into image space
            tmp = fft(tmp, axis=axis)

        return tmp

    def extract_subgrid_contrib_to_facet(self, FSi, facet_off, axis):
        """
        Extract contribution of subgrid to a facet.

        :param Fsi: Prepared subgrid in image space (see prepare_facet)
        :param facet_off: facet offset
        :param axis: axis along which the operations are performed (0 or 1)

        :return: Contribution of subgrid to facet

        """

        # Align with image zero in image space
        FSi = numpy.roll(
            FSi,
            -facet_off * self.xM_size // self.N,
            axis,
        )

        FNjSi = broadcast(self._Fn, len(FSi.shape), axis) * extract_mid(
            FSi,
            self.xM_yN_size,
            axis,
        )

        return numpy.roll(FNjSi, facet_off * self.xM_size // self.N, axis=axis)

    # pylint: disable=too-many-arguments
    def add_subgrid_contribution(
        self,
        subgrid_contrib,
        subgrid_off,
        axis,
    ):
        """
        Sum up subgrid contributions to a facet.

        :param subgrid_contrib: Subgrid contribution to this facet (see
                extract_subgrid_contrib_to_facet)
        :param subgrid_off: subgrid offset
        :param facet_m0_trunc: mask truncated to a facet (image space)
        :param axis: axis along which operations are performed (0 or 1)

        :return summed subgrid contributions

        """

        # Bring subgrid contribution into frequency space
        NjSi_temp = ifft(subgrid_contrib, axis)
        MiNjSi = numpy.roll(
            NjSi_temp,
            -subgrid_off * self.yN_size // self.N,
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
            subgrid_off * self.yN_size // self.N,
            axis=axis,
        )

    def finish_facet(
        self, MiNjSi_sum, facet_off, facet_size, facet_B_mask, axis
    ):
        """
        Obtain finished facet.

        It extracts from the padded facet (obtained from subgrid via FFT)
        the true-sized facet and multiplies with masked Fb.

        :param MiNjSi_sum: sum of subgrid contributions to a facet
        :param facet_size: facet size
        :param facet_off: facet offset
        :param facet_B_mask: a facet mask element
        :param axis: axis along which operations are performed (0 or 1)

        :return: finished (approximate) facet element
        """

        if facet_B_mask is not None:
            facet_mask = broadcast(
                extract_mid(self._Fb, facet_size, 0) * facet_B_mask,
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
            numpy.roll(fft(MiNjSi_sum, axis), -facet_off, axis=axis),
            facet_size,
            axis,
        )


class SwiftlyCoreFunc:
    """
    Streaming Distributed Fourier Transform class

    This version uses native implementations of SwiFTly functions from
    ska_sdp_func
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

        import ska_sdp_func.fourier_transforms.swiftly

        self._swiftly = ska_sdp_func.fourier_transforms.swiftly.Swiftly(
            N, yN_size, xM_size, W
        )

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

    def _auto_broadcast_create(
        self, create_fn, fn, in_arr, out_size, axis, out, *args
    ):
        # Convert to complex if needed
        if not numpy.iscomplexobj(in_arr):
            if in_arr.dtype == numpy.dtype(float):
                in_arr = in_arr.astype(complex)
            elif in_arr.dtype == numpy.dtype("float32"):
                in_arr = in_arr.astype("complex64")

        # One dimensional case: Add new axis to arrays
        if len(in_arr.shape) == 1:
            shape = (out_size,)
            if out is None:
                out = create_fn(shape, dtype=complex)
            elif out.shape != shape:
                raise ValueError(
                    f"Output array has shape {out.shape}, expected {shape}!"
                )
            fn(in_arr[numpy.newaxis], out[numpy.newaxis], *args)
            return out

        # Has to be two dimensional case now
        if len(in_arr.shape) != 2:
            raise ValueError(
                f"Invalid number of dimensions in input array: {in_arr}"
            )

        # Processing functions work on second axis, therefore transpose if we
        # are meant to work on the first axis.
        if axis == 0:
            shape = (out_size, in_arr.shape[1])
            if out is None:
                out = create_fn((out_size, in_arr.shape[1]), dtype=complex)
            elif out.shape != shape:
                raise ValueError(
                    f"Output array has shape {out.shape}, expected {shape}!"
                )

            fn(in_arr.T, out.T, *args)
            return out

        if axis == 1:
            shape = (in_arr.shape[0], out_size)
            if out is None:
                out = create_fn(shape, dtype=complex)
            elif out.shape != shape:
                raise ValueError(
                    f"Output array has shape {out.shape}, expected {shape}!"
                )
            fn(in_arr, out, *args)
            return out

        raise ValueError(f"Invalid axis {axis} for shape {in_arr.shape}!")

    def _auto_broadcast_create_2d(
        self, create_fn, fn, in_arr, out_size, out, *args
    ):
        # Convert to complex if needed
        if not numpy.iscomplexobj(in_arr):
            if in_arr.dtype == numpy.dtype(float):
                in_arr = in_arr.astype(complex)
            elif in_arr.dtype == numpy.dtype("float32"):
                in_arr = in_arr.astype("complex64")

        # Has to be two dimensional case now
        if len(in_arr.shape) != 2:
            raise ValueError(
                f"Invalid number of dimensions in input array: {in_arr}"
            )

        # Create or check output array
        shape = (out_size, out_size)
        if out is None:
            out = create_fn(shape, dtype=complex)
        else:
            assert (
                out.shape == shape
            ), "Output array has shape {out.shape}, expected {shape}!"

        fn(in_arr, out, *args)
        return out

    # facet to subgrid algorithm
    def prepare_facet(self, facet, facet_off, axis, out=None):
        """
        Prepare facet for extracting subgrid contribution

        This is a relatively expensive operation, both in terms of computation
        and generated data. It should therefore where possible be used for
        multiple :py:func:`extract_facet_contrib_to_subgrid` calls.

        :param facet: single facet element
        :param subgrid_off: subgrid offset
        :param axis: axis along which operations are performed (0 or 1)
        :return: prepared facet data
        """

        return self._auto_broadcast_create(
            numpy.empty,
            self._swiftly.prepare_facet,
            facet,
            self.yN_size,
            axis,
            out,
            facet_off,
        )

    def extract_facet_contrib_to_subgrid(
        self, prep_facet, subgrid_off, axis, out=None
    ):  # pylint: disable=too-many-arguments
        """
        Extract the facet contribution to a subgrid.

        :param prep_facet: prepared facet (see :py:func:`prepare_facet`)
        :param subgrid_off: subgrid offset
        :param axis: axis along which the operations are performed (0 or 1)

        :return: compact representation of contribution of facet to subgrid
        """

        return self._auto_broadcast_create(
            numpy.empty,
            self._swiftly.extract_from_facet,
            prep_facet,
            self.xM_yN_size,
            axis,
            out,
            subgrid_off,
        )

    extract_from_facet = extract_facet_contrib_to_subgrid

    def add_facet_contribution(self, facet_contrib, facet_off, axis, out=None):
        """
        Further transforms facet contributions, which then will be summed up.

        :param facet_contrib: array-chunk of individual facet contributions
        :param facet_off: facet offset for the facet_contrib array chunk
        :param axis: axis along which the operations are performed (0 or 1)
        :return: addition to subgrid
        """

        return self._auto_broadcast_create(
            numpy.zeros,
            self._swiftly.add_to_subgrid,
            facet_contrib,
            self.xM_size,
            axis,
            out,
            facet_off,
        )

    add_to_subgrid = add_facet_contribution

    def add_to_subgrid_2d(self, facet_contrib, facet_offs, out=None):
        """
        Further transforms facet contributions, which then will be summed up.

        :param facet_contrib: array-chunk of individual facet contributions
        :param facet_off: facet offset for the facet_contrib array chunk
        :param axis: axis along which the operations are performed (0 or 1)
        :return: addition to subgrid
        """

        return self._auto_broadcast_create_2d(
            numpy.zeros,
            self._swiftly.add_to_subgrid_2d,
            facet_contrib,
            self.xM_size,
            out,
            facet_offs[0],
            facet_offs[1],
        )

    def finish_subgrid(
        self, summed_contribs, subgrid_off, subgrid_size, subgrid_masks=None
    ):
        """
        Obtain finished subgrid. Operation performed for all axes.

        :param summed_contribs: summed facet contributions to this subgrid
        :param subgrid_off: subgrid offset per axis
        :param subgrid_size: subgrid size
        :param subgrid_masks: subgrid mask per axis (optional)
        :return: approximate subgrid element
        """

        if len(summed_contribs.shape) == 1:
            out = numpy.empty(subgrid_size, dtype=complex)
            self._swiftly.finish_subgrid(
                summed_contribs[numpy.newaxis], out[numpy.newaxis], subgrid_off
            )
            return out

        if len(summed_contribs.shape) == 2:
            if not isinstance(subgrid_off, list) or len(subgrid_off) != 2:
                raise ValueError(
                    "Subgrid offset must be given for every dimension!"
                )
            out1 = numpy.empty((self.xM_size, subgrid_size), dtype=complex)
            self._swiftly.finish_subgrid(summed_contribs, out1, subgrid_off[0])
            out = numpy.empty((subgrid_size, subgrid_size), dtype=complex)
            self._swiftly.finish_subgrid(out1.T, out.T, subgrid_off[1])
            return out

        raise ValueError(f"Invalid shape {summed_contribs.shape}!")

    # subgrid to facet algorithm
    def prepare_subgrid(self, subgrid, subgrid_off):
        """
        Calculate the FFT of a padded subgrid element.
        No reason to do this per-axis, so always do it for all axes.

        :param subgrid: single subgrid array element
        :param subgrid_off: subgrid offsets (tuple)

        :return: Padded subgrid in image space
        """

        # Convert to complex if needed
        if not numpy.iscomplexobj(subgrid):
            if subgrid.dtype == numpy.dtype(float):
                subgrid = subgrid.astype(complex)
            elif subgrid.dtype == numpy.dtype("float32"):
                subgrid = subgrid.astype("complex64")

        if len(subgrid.shape) == 1:
            subgrid = pad_mid(subgrid, self.xM_size, 0)
            self._swiftly.prepare_subgrid_inplace(
                subgrid[numpy.newaxis], subgrid_off
            )
            return subgrid

        elif len(subgrid.shape) == 2:
            if not isinstance(subgrid_off, list) or len(subgrid_off) != 2:
                raise ValueError(
                    "Subgrid offset must be given for every dimension!"
                )
            # TODO: Remove intermediate allocation
            subgrid = pad_mid(
                pad_mid(subgrid, self.xM_size, 0), self.xM_size, 1
            )
            self._swiftly.prepare_subgrid_inplace_2d(
                subgrid, subgrid_off[0], subgrid_off[1]
            )
            return subgrid

        raise ValueError(f"Invalid shape {subgrid.shape}!")

    def extract_subgrid_contrib_to_facet(self, FSi, facet_off, axis, out=None):
        """
        Extract contribution of subgrid to a facet.

        :param Fsi: Prepared subgrid in image space (see prepare_facet)
        :param facet_off: facet offset
        :param axis: axis along which the operations are performed (0 or 1)

        :return: Contribution of subgrid to facet

        """

        return self._auto_broadcast_create(
            numpy.empty,
            self._swiftly.extract_from_subgrid,
            FSi,
            self.xM_yN_size,
            axis,
            out,
            facet_off,
        )

    # pylint: disable=too-many-arguments
    def add_subgrid_contribution(
        self, subgrid_contrib, subgrid_off, axis, out=None
    ):
        """
        Sum up subgrid contributions to a facet.

        :param subgrid_contrib: Subgrid contribution to this facet (see
                extract_subgrid_contrib_to_facet)
        :param subgrid_off: subgrid offset
        :param facet_m0_trunc: mask truncated to a facet (image space)
        :param axis: axis along which operations are performed (0 or 1)

        :return summed subgrid contributions

        """

        return self._auto_broadcast_create(
            numpy.zeros,
            self._swiftly.add_to_facet,
            subgrid_contrib,
            self.yN_size,
            axis,
            out,
            subgrid_off,
        )

    def finish_facet(
        self, facet_sum, facet_off, facet_size, facet_B_mask, axis, out=None
    ):
        """
        Obtain finished facet.

        It extracts from the padded facet (obtained from subgrid via FFT)
        the true-sized facet and multiplies with masked Fb.

        :param MiNjSi_sum: sum of subgrid contributions to a facet
        :param facet_size: facet size
        :param facet_off: facet offset
        :param facet_B_mask: a facet mask element
        :param axis: axis along which operations are performed (0 or 1)

        :return: finished (approximate) facet element
        """

        return self._auto_broadcast_create(
            numpy.empty,
            self._swiftly.finish_facet,
            facet_sum,
            facet_size,
            axis,
            out,
            facet_off,
        )
