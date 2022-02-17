include .make/base.mk
include .make/python.mk

# W503: line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=W503