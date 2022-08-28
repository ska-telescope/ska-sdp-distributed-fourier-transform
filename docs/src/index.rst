.. _documentation_master:

.. toctree::

SKA SDP Distributed Fourier Transform
#####################################

This is a `repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform>`_
for the Dask-implemented distributed
Fourier transform algorithm used for radio interferometry processing.
It generates arbitrary grid chunks while minimising memory residency,
data transfer and compute work.

The code is written in Python. For details of the algorithm, please refer
to the :ref:`algorithm` page.


Installation Instructions
=========================

To install the repository, use git::

    git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform.git


Please ensure you have all the dependency packages installed. T
he installation is managed through `poetry <https://python-poetry.org/docs/>`_.
Refer to their page for instructions.

Running from the command line
=============================

A simple CLI is available to run the main python script, which executes the
Distributed FFT algorithm (facet->subgrid and subgrid->facet directions)::

    python -m src.ska_sdp_exec_swiftly.fourier_transform_dask --swift-config "<config-key>"

You need to replace `<config-key>` with one of the dictionary keys
found in src/switf_configs.py. Default is "1k[1]-n512-256".
The code will iterate through multiple configurations if you provide
the keys as coma-separated arguments `<config_key1>,<config_key2>`.

Specify the `DASK_SCHEDULER` environment variable to use a specific
Dask cluster for your run.

There are a few other scripts that help investigate the algorithm
for specific purposes and use cases. Please see :ref:`scripts` page
for details of their usage.

Running from the slurm cluster
=============================

We have provided some example slurm scripts to help run the program on slurm clusters.
Please refer to the slurm_scripts directory. The examples are based on two machines:
1. `CSD3 <https://docs.hpc.cam.ac.uk/hpc/>`_  at University of Cambridge, UK
2. AstroLab at Guangzhou University, China
You may need to modify them to suit the specific machine that you are using.

.. toctree::
   :maxdepth: 2

   algorithm
   examples
   structure
   api
   dask
   scripts
