. _algorithm:

Usage Examples
============

To test an image configuration (that is not the default TARGET_PARS), you need to specify the parameters as a Python dictionary with variables that are defined in the DistributedFFT class.
What need to be specified include:

 1. W:  PSWF parameter (grid-space support)
 2. fov:  field of view (in degrees)
 3. N: total image size
 4. Nx: subgrid spacing: it tells you what subgrid offsets are permissible.
 5. yB_size: true usable image size (facet)
 6. yN_size: padding needed to transfer the data?
 7. yP_size: padded (rough) image size (facet)
 8. xA_size: true usable subgrid size
 9. xM_size:  padded (rough) subgrid size

The parameters need to be consistent for the code to run. For details of the "parameter search", you can refer to the first half of facet-subgrid-impl.ipynb.

One configuration example is ::

 test_conf = "4k[1]-n2k-512":
 dict(W=11.0, fov=1, N=4096, Nx=64, yB_size=1408, yN_size=2048, yP_size=2048, xA_size=448, xM_size=512)

which stands for a 4k size image. You will need to replace that with the default in the main function in fourier_transform_2d_dask.py.