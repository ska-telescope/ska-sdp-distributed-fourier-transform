.. _documentation_master:

.. toctree::

ska-sdp-distributed-fourier-transform
###########################################################

This is a repository for the Dask-implemented distributed Fourier transform algorithm used for radio interferometry processing.
It generates arbitrary grid chunks with full non-coplanarity corrections while minimising memory residency, data transfer and compute work.

The code is written in python. For details of the algorithm, please refer to the "Description of the Algorithm" page.



Installation Instructions
============

To install the repository, use git::

    git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-distributed-fourier-transform.git


Please ensure you have all the dependency packages installed. The installation is managed through poetry (https://python-poetry.org/docs/).
Refer to their page for instructions.

.. toctree::
   :maxdepth: 2

   algorithm
   examples
   structure
