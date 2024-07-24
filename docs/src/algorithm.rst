
.. _algorithm:

Description of the Algorithm
============================

Introduction
------------

In Fourier transformation of imaging, every visibility impacts data of
every image pixel thus making it harder to implement distributed computation directly.
The distributed Fourier transform is designed in a way that it is a scalable method
that distributes the data efficiently without holding redundant image data,
which can be used for the FFT and gridding part of an imaging pipeline.

The key idea is using "facet-based" imaging. Since we are only interested in specific
sub-regions in grid space, we want to transfer arbitrary portions of frequency space
data to or from workers that are gridding and degridding visibilities, a.k.a. a
semi-sparse Fourier transform for chunks of image (time) and grid (frequency) space.

In short, the problem we are trying to solve is how to convert from image (time) space
to grid (frequency) space and vica versa in a distributed and efficient way which
does not involve having to keep the full image / grid in one place during the process.


Details of the Algorithm
------------------------

The mathematical derivation can be found in
`facet-subgrid.ipynb <https://gitlab.com/scpmw/crocodile/-/blob/io_benchmark/examples/notebooks/facet-subgrid.ipynb>`_.
A major part of implementing the algorithm is finding a suitable set of parameters,
which allow for an acceptable precision in reconstructing sub-grids from facets and vica versa.
The parameters define the relation between facet and sub-grid distribution,
i.e. how many facets and sub-grids we need to perform the calculations,
how big a single facet/sub-grid needs to be and what kind of padding we need in order to
safely convert between the two spaces.


We have implemented the algorithm in 2D. It starts with generating some random data
that is used to create a sub-grid and a facet array. The number of facets and sub-grids
in the array will depend on the total image size and some other size-parameters mentioned above.
These arrays will form the input to the algorithm but also act as comparison to the output of the algorithm.
The code performs calculations in two directions:

1) take the array of facets as input and obtain an array of approximate sub-grids

2) take the array of sub-grids as input and obtain an array of approximate facets

In an ideal situation, the approximate arrays would match their pre-generated equivalents
(i.e. approx_subgrid == subgrid and approx_facet == facet).

Previously, we have also implemented a 1D version with the same structure. Please checkout the
`tag version 0.0.1 <https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/tree/v0.0.1>`_
if you'd like to explore further the 1D cases.
