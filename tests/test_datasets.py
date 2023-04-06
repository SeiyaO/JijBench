from __future__ import annotations

import pytest

import jijbench as jb
from jijbench.datasets.model import InstanceDataFileStorage


@pytest.mark.parametrize(
    "problem_name",
    [
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ],
)
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_get_models(problem_name: str, size: str):
    n = 1
    models = jb.get_models(problem_name, size, n)

    assert len(models) == n

    problem, instance_data = models[0]

    assert problem.name == problem_name
    assert isinstance(instance_data, dict)


@pytest.mark.parametrize(
    "problem_name",
    [
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ],
)
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_get_names(problem_name: str, size: str):
    n = 1

    storage = InstanceDataFileStorage(problem_name)
    names = storage.get_names(size, n)

    assert len(names) == n
    assert isinstance(names[0], str)


@pytest.mark.parametrize(
    "problem_name",
    [
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ],
)
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_get_files_map(problem_name: str, size: str):
    n = 1

    storage = InstanceDataFileStorage(problem_name)
    names = storage.get_names(size, n)
    files = storage.get_files(size, n)
    files_map = storage.get_files_map(size, n)

    for name, file in zip(names, files):
        assert name in files_map
        assert files_map[name] == file
