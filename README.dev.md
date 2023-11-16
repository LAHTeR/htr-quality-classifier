# `text_quality` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

```shell
# Create a virtual environment, e.g. with
python3 -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python3 -m pip install --upgrade pip setuptools

# (from the project root directory)
# install text_quality as an editable package
python3 -m pip install --no-cache-dir --editable .
# install development dependencies
python3 -m pip install --no-cache-dir --editable .[dev]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `text_quality`, but then builds the necessary virtual environments itself by simply running:

```shell
tox
```

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Running linters locally

For linting we will use [prospector](https://pypi.org/project/prospector/) and to sort imports we will use
[isort](https://pycqa.github.io/isort/). Running the linters requires an activated virtual environment with the
development tools installed.

```shell
# linter
prospector

# recursively check import style for the text_quality module only
isort --check-only text_quality

# recursively check import style for the text_quality module only and show
# any proposed changes as a diff
isort --check-only --diff text_quality

# recursively fix import style for the text_quality module only
isort text_quality
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf).

You can enable automatic linting with `prospector` and `isort` on commit by enabling the git hook from `.githooks/pre-commit`, like so:

```shell
git config --local core.hooksPath .githooks
```

## Generating the Architecture Diagram

The architecture diagram is stored in the [classes_text_quality.svg](classes_text_quality.svg) file, and displayed in the [README.md](README.md) file.
To update it, use [pyreverse](https://pylint.readthedocs.io/en/latest/pyreverse.html) from the [pylint](https://pypi.org/project/pylint/) package:

```shell
pyreverse --output svg --project text_quality text_quality
```

## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`

If you do not have `make` use

```shell
sphinx-build -b html docs docs/_build/html
```

To find undocumented Python objects run

```shell
cd docs
make coverage
cat _build/coverage/python.txt
```

To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in documentation run

```shell
cd docs
make doctest
```

## Versioning

Bumping the version across all files is done with [bumpversion](https://github.com/c4urself/bump2version), e.g.

```shell
bumpversion major
bumpversion minor
bumpversion patch
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Make sure the [version has been updated](#versioning).
4. Run the unit tests with `pytest -v`

### SKIP: (2/3) PyPI Release

Publishing an updated package on PyPI manually is not necessary for this project.
Instead, the [Build and Publish Workflow](.github/workflows/publish-to-test-pypi.yml) is triggered automatically when a new release is created on GitHub in the [next step](#33-github).

### (3/3) GitHub Release

Make a [release on GitHub](https://github.com/laHTeR/htr-quality-classifier/releases/new).
Create a new tag in the form `v<X.X.X>`, where `<X.X.X>` is the version number as specified in the [versioning section](#versioning).

This will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it (see [Zenodo project page](https://zenodo.org/doi/10.5281/zenodo.8189892)).
