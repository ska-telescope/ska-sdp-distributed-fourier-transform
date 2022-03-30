# SKA SDP Distributed Fourier Transform

This is a repository for the Dask implemented distributed Fourier transform algorithm used for radio interferometry processing. Please refer to the documentation for details and installation instructions.

## Standard CI machinery

This repository is set up to use the
[Makefiles](https://gitlab.com/ska-telescope/sdi/ska-cicd-makefile) and [CI
jobs](https://gitlab.com/ska-telescope/templates-repository) maintained by the
System Team. For any questions, please look at the documentation in those
repositories or ask for support on Slack in the #team-system-support channel.

To keep the Makefiles up to date in this repository, follow the instructions
at: https://gitlab.com/ska-telescope/sdi/ska-cicd-makefile#keeping-up-to-date

## Contributing to this repository

[Black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/),
and various linting tools are used to keep the Python code in good shape.
Please check that your code follows the formatting rules before committing it
to the repository. You can apply Black and isort to the code with:

```bash
make python-format
```

and you can run the linting checks locally using:

```bash
make python-lint
```

The linting job in the CI pipeline does the same checks, and it will fail if
the code does not pass all of them.

