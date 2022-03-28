
Fourier algorithm with Dask
===========================

`Dask <https://dask.org/>`_ is used for the parallelization of the algorithm
allowing for running it in a distributed manner.

We created a wrapper, which, if conditions met, will execute the
wrapped function using dask.delayed. Else, it will execute the function in serial.

Dask.delayed processes the functions wrapped with it, and build
a graph, which is meant to optimize function execution and make sure that
the same call is not processed more than needed (keeping its result in memory
and passing it down to the functions that need it). The graph is
optimized to executes calls in parallel where possible.

We use dask.distributed to create a Client object, which connects
to a Scheduler. The Scheduler distributes the work between Workers.
At the moment, the Client sets up a Dask Cluster using its default,
which is to take into account the available cores in the machine that
runs the code. In the future, we can update the code to make the Client
customizable and allow for specifying a scheduler.
