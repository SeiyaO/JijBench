from __future__ import annotations

import os
import shutil
import typing as tp
from unittest.mock import MagicMock

import jijmodeling as jm
import pandas as pd
import pytest

import jijbench as jb
from jijbench.typing import ModelType


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def f1():
    return "Hello, World!"


def f2(i: int) -> int:
    return i**2


def f3(i: int) -> tuple[int, int]:
    return i + 1, i + 2


def f4(i: int, j: int = 1) -> int:
    return i + j


def f5(model: jm.Problem, feed_dict: jm.PH_VALUES_INTERFACE) -> ModelType:
    return model, feed_dict


def test_sample(problem):
    print(problem)


def test_simple_experiment():
    e = jb.Experiment(name="simple_experiment")
    solver = jb.Solver(f1)
    record = solver([])
    e.append(record)
    e.save()

    print(e.table)


@pytest.mark.parametrize(
    "solver, params",
    [
        (f1, []),
        (f2, [{"i": 1}, {"i": 2}, {"i": 3}]),
        (f3, [{"i": 1}, {"i": 2}, {"i": 3}]),
        (f4, [{"i": 1}, {"i": 1, "j": 2}, {"i": 1, "j": 3}]),
    ],
)
def test_experiment(
    solver: tp.Callable[..., tp.Any], params: dict[str, tp.Iterable[tp.Any]]
):
    e = jb.Experiment(name="simple_experiment")

    for param in params:
        solver = jb.Solver(solver)
        record = solver([jb.Parameter(i, "i") for k, v in param.items() for i in v])
        e.append(record)
    e.save()


def test_simple_experiment_with_context_manager():
    e = jb.Experiment(name="simple_experiment_with_context_manager", autosave=True)

    def func(i):
        return i**2

    for i in range(3):
        with e:
            solver = jb.Solver(func)
            record = solver([jb.Parameter(i, "i")])
            e.append(record)


def test_jijmodeling(
    knapsack_problem: jm.Problem,
    knapsack_instance_data: jm.PH_VALUES_INTERFACE,
    jm_sampleset: jm.SampleSet,
):
    def func(model, feed_dict):
        return jm_sampleset

    experiment = jb.Experiment(autosave=False)

    with experiment:
        solver = jb.Solver(func)
        x1 = jb.Parameter(knapsack_problem, name="model")
        x2 = jb.Parameter(knapsack_instance_data, name="feed_dict")
        record = solver([x1, x2])
        record.name = jb.ID().data
        experiment.append(record)

    droped_table = experiment.table.dropna(axis="columns")

    cols = droped_table.columns
    assert "energy" in cols
    assert "num_feasible" in cols
