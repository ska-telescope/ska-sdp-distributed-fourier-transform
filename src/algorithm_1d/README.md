Note: This file contains general information about the Distributed FFT algorithm,
in order to give a context to its 1D version archived in the current (`algorithm_1d`) directory.

## Introduction 

In Fourier transformation of imaging, every visibility impacts data of every image pixel 
thus making it harder to implement distributed computation directly. 
The distributed Fourier transform is designed in a way that, it is a scalable method that 
distributes the data efficiently without holding redundant image data, which can be used for the FFT 
and gridding part of an imaging pipeline.

The key idea is using "facet-based" imaging. Since we are only interested a specific 
sub-regions in grid space, we want to transfer virtually arbitrary portions of frequency space 
data to or from workers that are gridding and degridding visibilities, aka a semi-sparse Fourier transform 
for chunks of image(time) and grid(frequency) space.

## Description of Algorithm

Detailed problem definition and algorithm derivation can be found in 
[facet-subgrid.ipynb](https://gitlab.com/scpmw/crocodile/-/blob/io_benchmark/examples/notebooks/facet-subgrid.ipynb). 
In short, the problem we are trying to solve is how to convert from image (time) space 
to grid (frequency) space and vica versa in a distributed and efficient way which does 
not involve having to keep the full image / grid in one place during the process.

The mathematical derivation can be found in facet-subgrid.ipynb. 
A major part of implementing the algorithm is finding a suitable set of parameters, 
which allow for an acceptable precision in reconstructing sub-grids from facets and vica versa. 
The parameters define the relation between facet and sub-grid distribution, 
i.e. how many facets and sub-grids we need to perform the calculations, 
how big a single facet/sub-grid needs to be and what kind of padding we need 
in order to safely convert between the two spaces. The details of the "parameter search" 
are in the first half of facet-subgrid-impl.ipynb.

## Implementation

We have implemented two main versions based on the notebooks: 1D and 2D. 
Both start with generating some random data that is used to create a sub-grid and a facet array. 
The number of facets and sub-grids in the array will depend on the total image size and 
some other size-parameters mentioned above. These arrays will form the input to the algorithm 
but also act as comparison to the output of the algorithm. Both 1D and 2D versions perform 
calculations in two directions:

- take the array of facets as input and obtain an array of approximat sub-grids 
- take the array of sub-grids as input and obtain an array of approximate facets

In an ideal situation, the approximate arrays would match their pre-generated equivalents 
(i.e. approx_subgrid == subgrid and approx_facet == facet).

### 1D algorithm

This directory (`algorithm_1d`) contains all the relevant code that we developed
and worked on for the 1D version of the distributed FFT algorithm. This version
was intended as an introduction to the algorithm and not as a final version of the code.

In a future commit, we are going to remove this directory from main,
and only keep the 2D and more general version of the code for simplicity and clarity.

## Further reading & resources

- Peter Wortmann's paper: https://arxiv.org/abs/2108.10720
- LOFAR facet calibration: https://ui.adsabs.harvard.edu/abs/2016ApJS..223....2V/abstract, https://ui.adsabs.harvard.edu/abs/2018A%26A...611A..87T/abstract
- Tim Cornwell's 1992 paper on faceting: https://ui.adsabs.harvard.edu/abs/1992A%26A...261..353C/abstract
- Facet-based imaging in WSClean: https://wsclean.readthedocs.io/en/latest/facet_based_imaging.html

