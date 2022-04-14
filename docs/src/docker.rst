
Docker image
============

A Docker image is also available, which contains the full
contents of the repository, and it has all the requirements installed.

It is available from the
`Central Artifact Repository <https://artefact.skao.int/#browse/browse:docker-all>`_::

    artefact.skao.int/ska-sdp-distributed-fourier-transform

Please refer to the repository for the latest version, and other available versions.
The first version we published is 0.1.0. Note that there is a 0.0.1 tag on the
GitLab repository, however, that does not have a corresponding Docker image.

The image is based on `python:3.9-slim` and its entrypoint has not been updated.
Running the Docker image will start a python:3.9 shell.

In order to access the scripts and run them from bash, execute the following::

    docker run -it artefact.skao.int/ska-sdp-distributed-fourier-transform:0.1.0 /bin/bash

Make sure you use the correct tag. It will take you to the `/app` directory,
which contains all the files from the repository.

You may also use jupyter-lab, which is also installed on the image (only from v0.2.0).
