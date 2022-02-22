include .make/base.mk
include .make/python.mk

python-do-lint:
	@echo "Temporary commenting linting"

python-pre-test:  ## Install requirements-test.txt
	poetry config virtualenvs.in-project true

PYTHON_SWITCHES_FOR_FLAKE8 = --ignore=E501,F405,W503,E203,F403,F401,E266
