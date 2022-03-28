include .make/base.mk
include .make/python.mk
include .make/docs.mk

python-do-lint:
	black --check src/ tests/
