include .make/base.mk
include .make/python.mk

python-pre-lint:
	pip3 install black

python-do-lint:
	black --check src/ tests/

#PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=E501,F405,W503,E203,F403,F401,E266
