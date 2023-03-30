
# JijBench: An Experiment and Benchmark Management Library for Mathematical Optimization

JijBench is a Python library designed for developers working on research and development or proof-of-concept experiments using mathematical optimization. Positioned similarly to mlflow in the machine learning field, JijBench provides features such as saving optimization results, automatically computing benchmark metrics, and offering visualization tools for the results.

Primarily supporting Ising optimization problems, JijBench plans to extend its support to a wide range of optimization problems, such as MIP solvers, in the future.

## Installation
JijBench can be easily installed using pip.

``` shell
pip install jijbench
```

## Documentation and Support
Tutorials and sample code will be provided in the future. Stay tuned!



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
```
pre-commit install
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

