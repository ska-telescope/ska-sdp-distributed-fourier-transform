include .make/base.mk
include .make/python.mk
include .make/docs.mk
include .make/oci.mk

PROJECT_NAME = ska-sdp-distributed-fourier-transform

python-do-lint:
	black --check src/ tests/
