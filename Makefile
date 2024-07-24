include .make/base.mk
include .make/python.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-exec-swiftly

# C0103: invalid-name (caused by non-compliant variable names)
# W0511: fixme (don't report TODOs)
# R0801: duplicate-code (some are duplicated between the main function and utils
# R0914: Too many local variables
# R0915: Too many statements
#		 these will eventually need to be resolved, utils cleaned up
PYTHON_SWITCHES_FOR_PYLINT = --disable=C0103,W0511,R0801,R0914,R0915

# E203: flake8 and black don't agree on "extra whitespace", we ignore the flake8 error
# W503: same problem with "line break before binary operator"
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=E203,W503
PYTHON_LINT_TARGET = src/ tests/ scripts/

# Ignore sins in notebook (it is mostly for historical interest anyway)
# A lot of them seem to be just plain wrong?
NOTEBOOK_SWITCHES_FOR_PYLINT = --disable=E0602,C0116,W0621,E0102,C0413,C0209,W0122,R0913,R1705,E0601,C0301,C0114,R0402,R1728
NOTEBOOK_SWITCHES_FOR_FLAKE8 = --ignore=E501,F821,F811,W503,E203,E402
