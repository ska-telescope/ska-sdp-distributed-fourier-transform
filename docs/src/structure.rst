. _structure:

Code Structure and APIs
============

Data Models
--------------------------------------

The main data classes are:

.. automodapi::    src.fourier_transform.algorithm_parameters
   :no-inheritance-diagram:



Base Functions
--------------------------------------

.. automodapi::    src.fourier_transform.fourier_algorithm
   :no-inheritance-diagram:


Main Processing Functions (2D) with Dask
--------------------------------------

The dask wrapper:

We wrap the Dask delayed implementation in a dask wrapper where if use_dask is set to True, the dask wrapper function will call the dask.delayed option for the computation.

.. automodapi::    src.fourier_transform.dask_wrapper
   :no-inheritance-diagram:

The functions that conduct the main dask implemented algorithm include the subgrid to facet, and facet to subgrid transformations.
They are listed below. The main function calls all the functions.

.. automodapi::    src.fourier_transform_2d_dask
   :no-inheritance-diagram:

Checking the Results and Plotting
--------------------------------------

.. automodapi::    src.utils
   :no-inheritance-diagram:
