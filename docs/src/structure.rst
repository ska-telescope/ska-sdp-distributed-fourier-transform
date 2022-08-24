
Main Repository Structure
====================

The following diagram demonstrates the folder structure in the
`repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform>`_::

    ska-sdp-distributed-fourier-transform
    |__ .make
    |__ docs
    |__ notebook
    |__ scripts
    |__ slurm_scripts
    |__ src
    |   |__ fourier_transform
    |           algorithm_parameters.py
    |           fourier_algorithm.py
    |           dask_wrapper.py
    |       fourier_transform_dask.py
    |       utils.py
    |       swift_configs.py
    |       api.py
    |       api_helper.py
    |       generate_hdf5.py
    |__ tests
        <repository-related-files; e.g. poetry.lock, gitignore, Makefile, etc>

- **.make:** `ska-cicd-makefile <https://gitlab.com/ska-telescope/ska-cicd-makefile>`_ submodule

- **docs:** contains documentation related files

- **notebook:** contains a Jupyter notebook, which is a shortened version of facet-subgrid-impl.ipynb.
  It only contains the algorithm and related functions but not the parameter search

- **src/fourier_transform:** contains all the relevant functions and classes

    * **algorithm_parameters.py:** base classes for data models and the actual algorithm

    * **fourier_algorithm.py:** contains functions commonly used by multiple parts of the algorithm

    * **dask_wrapper.py:** contains a few functions that help wrap the distributed coded into dask delayed,
      and set up and tear down the dask client.

- **src/fourier_transform_dask.py:** contains the main function which orchestrates the code.
  It also decides whether to run the code with Dask or not. It is set up to run the algorithm in 2D.

- **src/utils.py:** contains plotting and validation testing utils

- **src/swift_configs.py:** contains a dictionary of example configurations, which can be directly used with the code

- **src/generate_hdf5.py:** contains functions that involve generating data and storing them using HDF5 data format.
  This is used in cases where the initial data size is too big for the machine to handle.

- **src/api.py:** contains the basic data structure of the algorithm for the object-oriented API.
  Includes core parameters, the facet and subgrid class.

- **src/api_helper.py:** contains several helper functions for the API implementation.

Please refer to the :ref:`api` page for the details of functions and classes of the algorithm.

Please refer to the :ref:`scripts` page for the details of custom scripts.
