include .make/base.mk
include .make/python.mk
include .make/docs.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-distributed-fourier-transform

# C0103: invalid-name (caused by non-compliant variable names)
# W0511: fixme (don't report TODOs)
# R0801: duplicate-code (some are duplicated between the main function and utils
#		 these will eventually need to be resolved, utils cleaned up
PYTHON_SWITCHES_FOR_PYLINT = --disable=C0103,W0511,R0801
