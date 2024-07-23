
.. _scripts:

Custom Scripts
==============

We have provided several custom scripts to perform analysis on special components regardin the algorithm.
They are individual command line apps that take command line arguments.
Details of how to run each of them as follows.


API Demo
++++++++

To better support streaming computing using Distributed Fourier Transform, an object-oriented new API was
proposed and implemented. The API has the following features.
1. The core parameters and the core algorithm that do not rely on Dask implementation or take assumptions about the facet and subgrid layout
2. Descriptions of facets / subgrids. Each would have sizes, offsets and potentially a mask.

.. argparse::
   :filename: scripts/demo_api.py
   :func: dfft_parser
   :prog: demo_api.py

Sparse Facet Demo
++++++++++++++++++++++

Based on the support of Object-Oriented API, the script demonstrates the exciting feature of sparse facet
in Distributed Fourier Transform that could support the Fourier Transform of arbitrary facet in
some specific configurations.

.. argparse::
   :filename: scripts/demo_sparse_facet.py
   :func: dfft_parser
   :prog: demo_sparse_facet.py
