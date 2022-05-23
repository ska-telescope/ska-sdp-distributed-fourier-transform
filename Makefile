include .make/base.mk
include .make/python.mk
include .make/docs.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-distributed-fourier-transform

# C0103: invalid-name (caused by non-compliant variable names)
# W0511: fixme (don't report TODOs)
# R0801: duplicate-code (some are duplicated between the main function and utils
# R0914: Too many local variables
#		 these will eventually need to be resolved, utils cleaned up
PYTHON_SWITCHES_FOR_PYLINT = --disable=C0103,W0511,R0801,R0914

# E203: flake8 and black don't agree on "extra whitespace", we ignore the flake8 error
# W503: same problem with "line break before binary operator"
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=E203,W503
