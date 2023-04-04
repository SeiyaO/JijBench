# Getting started

Welcome to the Getting Started guide for [JijBench], an open-source Python library for managing benchmarking in mathematical optimization.
This guide will help you get started with [JijBench], and will walk you through the process of installing [JijBench] and running a benchmarking.

  [JijBench]: https://github.com/Jij-Inc/JijBench

## Installation

JijBench can be installed using `pip` or if you use poetry, you can add JijBench to your `pyproject.toml` file.

=== "pip"

    ``` sh
    pip install jijbench
    ```

=== "poetry"

    ``` sh
    poetry add jijbench
    ```

## Simple example

Here is a simple example of how to use [JijBench].

You can choose any function to benchmark. For this example, we'll use a function that returns the square of a number.

```python
def square(x):
    return x*x

```

The code below sets up the benchmark:

```python
import jijbench as jb

bench = jb.Benchmark(
    solver=square,
    params={'x': range(10)},
    name="example"
)

```

The constructor, `jb.Benchmark`, accepts three inputs:

`solver`: the function you want to benchmark

`params`: the parameters you want to benchmark

`name`: the name of the benchmark

You can experiment with multiple functions and arguments by providing them as tuples to `solver` and `params`.

We have finished setting up the benchmark. Let's run it!

```python
experiment_result = bench(autosave=True)

```

We can use autosave to save the benchmark result or not. All the benchmark results are stored in experiment_result. You can check the result of your experiment as a pandas.DataFrame with the table method.

```python
experiment_result.table

```

![](assets/images/getting_started1.png)

You can load the saved experiment result using jb.load.

```python
loaded_result = jb.load('example')
loaded_result.table

```

![](assets/images/getting_started2.png)