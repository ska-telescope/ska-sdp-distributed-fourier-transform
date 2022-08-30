Custom Scripts
===========================

We have provided several custom scripts to perform analysis on special components regardin the algorithm.
They are individual command line apps that take command line arguments.
Details of how to run each of them as follows.

Memory Consumption Analysis
++++++++++++++++++++++

The script is used to investigate the memory consumption for
the fundamental Distributed Fourier Transform implementation.

.. argparse::
   :filename: ../../scripts/memory_consumption.py
   :func: cli_parser
   :prog: memory_consumption.py

Performance Queue
++++++++++++++++++++++

The script is used to investigate the computing performance and memory consumption for
the improved Distributed Fourier Transform implementation that has queue support. It should be
noted that ths script only implemented Facet->Subgrid.

.. argparse::
   :filename: ../../scripts/performance_queue.py
   :func: cli_parser
   :prog: performance_queue.py

Performance Full Steps
++++++++++++++++++++++

The script is used to investigate the computing performance and memory consumption for
the Improved Distributed Fourier Transform implementation that has queue support. The
script is full function test script that supports Facet->Subgrid and Subgrid->Facet.

.. argparse::
   :filename: ../../scripts/performance_full_steps.py
   :func: cli_parser
   :prog: performance_full_steps.py


API Demo
++++++++++++++++++++++

To better support streaming computing using Distributed Fourier Transform, an object-oriented new API was
proposed and implemented. The API has the following features.
1. The core parameters and the core algorithm that do not rely on Dask implementation or take assumptions about the facet and subgrid layout
2. Descriptions of facets / subgrids. Each would have sizes, offsets and potentially a mask.

.. argparse::
   :filename: ../../scripts/demo_api.py
   :func: cli_parser
   :prog: demo_api.py

Sparse Facet Demo
++++++++++++++++++++++

Based on the support of Object-Oriented API, the script demonstrates the exciting feature of sparse facet
in Distributed Fourier Transform that could support the Fourier Transform of arbitrary facet in
some specific configurations.

.. argparse::
   :filename: ../../scripts/demo_sparse_facet.py
   :func: cli_parser
   :prog: demo_sparse_facet.py