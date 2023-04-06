
# JijBench: An Experiment and Benchmark Management Library for Mathematical Optimization

[![PyPI version shields.io](https://img.shields.io/pypi/v/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![PyPI implementation](https://img.shields.io/pypi/implementation/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![PyPI format](https://img.shields.io/pypi/format/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![PyPI license](https://img.shields.io/pypi/l/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![PyPI download month](https://img.shields.io/pypi/dm/jijbench.svg)](https://pypi.python.org/pypi/jijbench/)
[![Downloads](https://pepy.tech/badge/jijbench)](https://pepy.tech/project/jijbench)

[![Python Test](https://github.com/Jij-Inc/JijBench/actions/workflows/python-test.yml/badge.svg)](https://github.com/Jij-Inc/JijBench/actions/workflows/python-test.yml)
[![CodeQL](https://github.com/Jij-Inc/JijBench/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/Jij-Inc/JijBench/actions/workflows/github-code-scanning/codeql)
[![Upload Python Package](https://github.com/Jij-Inc/JijBench/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Jij-Inc/JijBench/actions/workflows/python-publish.yml)
[![codecov](https://codecov.io/gh/Jij-Inc/JijBench/branch/main/graph/badge.svg?token=pfEmtaSP8Z)](https://codecov.io/gh/Jij-Inc/JijBench)


JijBench is a Python library designed for developers working on research and development or proof-of-concept experiments using mathematical optimization. Positioned similarly to mlflow in the machine learning field, JijBench provides features such as saving optimization results, automatically computing benchmark metrics, and offering visualization tools for the results.

Primarily supporting Ising optimization problems, JijBench plans to extend its support to a wide range of optimization problems, such as MIP solvers, in the future.

## Installation
JijBench can be easily installed using pip.

``` shell
pip install jijbench
```

## Documentation and Support

Documentation: https://jij-inc.github.io/JijBench/

Tutorials will be provided in the future. Stay tuned!


## How to Contribute

> *Development Environment Policy*:  
> Our policy is to establish a simple development environment that allows everyone to easily contribute to the project. With this in mind, we carefully select the necessary commands for setting up the environment to be as minimal as possible. Based on this policy, we have adopted an environment using `poetry` in this project.

### Setup environment with `poetry`

1: Setup poetry
```
pip install -U pip
pip install poetry
poetry self add "poetry-dynamic-versioning[plugin]"
poetry install
```

2: Setup `pre-commit`

In this project, we use pre-commit hooks to help maintain code quality. This ensures that predefined checks and formatting are automatically executed before each commit.

`pre-commit` was installed by the above command `poetry install`.
So, next enable the pre-commit hooks by running the following command in the project's root directory:

```
pre-commit install
```

> **Notes on Using pre-commit:**  
> With pre-commit enabled, predefined checks and formatting will be automatically executed before each commit. If any errors are detected during this process, the commit will be aborted. You will not be able to commit until the errors are resolved, so please fix the errors and try committing again.

You may need run `black` and `isort` before commit.
```
python -m isort ./jijbench
python -m black ./jijbench
```

3: Check tests

```
poetry shell
python -m pytest tests
```

### When you want add a dependency

**Standard dependency**
```
poetry add ...
```

**Depencency for test**
```
poetry add ... -G tests
```

**Depencency for dev**
```
poetry add ... -G dev
```

---

Copyright (c) 2023 Jij Inc.

