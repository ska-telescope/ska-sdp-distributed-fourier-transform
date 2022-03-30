include .make/base.mk
include .make/python.mk
include .make/docs.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-distributed-fourier-transform

# C0103: invalid-name (caused by non-compliant variable names)
PYTHON_SWITCHES_FOR_PYLINT = --disable C0103
