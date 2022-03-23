
Usage Examples
============

To test an image configuration, you need to specify the parameters as a Python dictionary with variables that are defined in the DistributedFFT class.
The following parameters need to be included in the dictionary (see parameters in the BaseParameters class) :

 :py:class:`src.fourier_transform.algorithm_parameters.BaseParameters`

The parameters need to be consistent for the code to run. For details of the "parameter search", you can refer to the first half of the Python notebook:
`facet-subgrid-impl.ipynb <https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform/-/blob/main/notebook/facet-subgrid-impl.ipynb>`_

One configuration example is ::

 test_conf =
 dict(W=11.0, fov=1, N=4096, Nx=64, yB_size=1408, yN_size=2048, yP_size=2048, xA_size=448, xM_size=512)

which stands for a 4k size image. You will need to replace that with the default in the main function in fourier_transform_2d_dask.py.

There are also the options to specify whether to use Dask or not in the code, and whether to do a plotting check. Simply set::

  main(to_plot=True, use_dask=True)

If you would like to do a plotting test/use Dask.