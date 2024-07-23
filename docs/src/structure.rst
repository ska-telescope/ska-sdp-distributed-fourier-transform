Main Repository Structure
=========================

The following diagram demonstrates the folder structure in the
`repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform>`_::

    ska-sdp-distributed-fourier-transform
    |__ .make
    |__ docs
    |__ notebook
    |__ scripts
    |__ slurm_scripts
    |__ src
    |   |__ska_sdp_exec_swiftly
    |       |__fourier_transform
    |             algorithm_parameters.py
    |             fourier_algorithm.py
    |       swift_configs.py
    |       api.py
    |       api_helper.py
    |__ tests
        <repository-related-files; e.g. poetry.lock, gitignore, Makefile, etc>

- **.make:** `ska-cicd-makefile <https://gitlab.com/ska-telescope/ska-cicd-makefile>`_ submodule

- **docs:** contains documentation related files

- **notebook:** contains a Jupyter notebook, which is a shortened version of facet-subgrid-impl.ipynb.
  It only contains the algorithm and related functions but not the parameter search

- **src/ska_sdp_exec_swiftly:** contains all the relevant functions and classes

- **src/ska_sdp_exec_swiftly/fourier_transform** contains all the relevant functions and classes of fundamental fourier transform
    * **algorithm_parameters.py:** base classes for data models and the actual algorithm

    * **fourier_algorithm.py:** contains functions commonly used by multiple parts of the algorithm



- **src/ska_sdp_exec_swiftly/swift_configs.py:** contains a dictionary of example configurations, which can be directly used with the code

- **src/ska_sdp_exec_swiftly/api.py:** contains the basic data structure of the algorithm for the object-oriented API.
  Includes core parameters, the facet and subgrid class.

- **src/ska_sdp_exec_swiftly/api_helper.py:** contains several helper functions for the API implementation.

Please refer to the :ref:`api` page for the details of functions and classes of the algorithm.

Please refer to the :ref:`scripts` page for the details of custom scripts.

