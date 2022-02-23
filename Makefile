include .make/base.mk
include .make/python.mk

python-do-lint:
	black --check src/ tests/
