. _algorithm:

Description of the Algorithm
###########################################################

Introduction
++++++++++++++++++++

In Fourier transformation of imaging, every visibility impacts data of every image pixel thus making it harder to implement distributed computation directly. The distributed Fourier transform is designed in a way that, it is a scalable method that distributes the data efficiently without holding redundant image data, which can be used for the FFT and gridding part of an imaging pipeline.
The key idea is using "facet-based" imaging. Since we are only interested a specific sub regions in grid space, we want to transfer virtually arbitrary portions of frequency space data to or from workers that are gridding and degridding visibilities, aka a semi-sparse Fourier transform for chunks of image(time) and grid(frequency) space.

Details of the Algorithm
++++++++++++++++++++

The mathematical derivation can be found in facet-subgrid.ipynb.
A major part of implementing the algorithm is finding a suitable set of parameters, which allow for an acceptable precision in reconstructing sub-grids from facets and vica versa.
The parameters define the relation between facet and sub-grid distribution, i.e. how many facets and sub-grids we need to perform the calculations, how big a single facet/sub-grid needs to be and what kind of padding we need inorder to safely convert between the two spaces.
The details of the "parameter search" are in the first half of facet-subgrid-impl.ipynb.

We have implemented the algorithm in both 1D and 2D.
Both start with generating some random data that is used to create a sub-grid and a facet array. The number of facets and sub-grids in the array will depend on the total image size and some other size-parameters mentioned above.
These arrays will form the input to the algorithm but also act as comparison to the output of the algorithm.
Both 1D and 2D versions perform calculations in two directions:

1) take the array of facets as input and obtain an array of approximat sub-grids

2) take the array of sub-grids as input and obtain an array of approximate facets

In an ideal situation, the approximat arrays would match their pre-generated equivalents (i.e. approx_subgrid == subgrid and approx_facet == facet).