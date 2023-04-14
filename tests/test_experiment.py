from __future__ import annotations

import os
import shutil
import typing as tp

import jijmodeling as jm
import pandas as pd
import pytest

import jijbench as jb
from jijbench.experiment.experiment import _Created, _Running, _Waiting


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


@pytest.mark.parametrize(
    "f, param_names, param_values",
    [
        (f1, [(), (), ()], [(), (), ()]),
        (f2, ("i",), [(1,), (2,), (3,)]),
        (f3, ("i",), [(1,), (2,), (3,)]),
        (f4, ("i", "j"), [(1,), (1, 2), (1, 3)]),
    ],
)
def test_experiment(
    f: tp.Callable[..., tp.Any],
    param_names: tuple[str, ...],
    param_values: list[tuple[int, ...]],
):
    experiment = jb.Experiment(name="simple_experiment")

    state = getattr(experiment, "state")
    assert isinstance(state, _Created)

    solver = jb.Solver(f)
    for i, v in enumerate(param_values):
        record = solver([jb.Parameter(vi, name) for name, vi in zip(param_names, v)])
        experiment.append(record)

        actual = experiment.table.filter(regex="solver_return").tail(1)
        expected = pd.DataFrame([f(*v)], index=actual.index, columns=actual.columns)

        assert actual.index[-1] == i
        assert actual.equals(expected)

        state = getattr(experiment, "state")
        assert isinstance(state, _Waiting)
    experiment.save()


@pytest.mark.parametrize(
    "f, param_names, param_values",
    [
        (f1, [(), (), ()], [(), (), ()]),
        (f2, ("i",), [(1,), (2,), (3,)]),
        (f3, ("i",), [(1,), (2,), (3,)]),
        (f4, ("i", "j"), [(1,), (1, 2), (1, 3)]),
    ],
)
def test_experiment_with_context_manager(
    f: tp.Callable[..., tp.Any],
    param_names: tuple[str, ...],
    param_values: list[tuple[int, ...]],
):
    experiment = jb.Experiment(name="simple_experiment")

    state = getattr(experiment, "state")
    assert isinstance(state, _Created)

    solver = jb.Solver(f)
    for v in param_values:
        with experiment:
            record = solver(
                [jb.Parameter(vi, name) for name, vi in zip(param_names, v)]
            )
            experiment.append(record)

            actual = experiment.table.filter(regex="solver_return").tail(1)
            expected = pd.DataFrame([f(*v)], index=actual.index, columns=actual.columns)

            assert actual.index.get_level_values(0)[-1] == experiment.name
            assert actual.equals(expected)

            state = getattr(experiment, "state")
            assert isinstance(state, _Running)
    experiment.save()


def test_experiment_with_solver_returning_jm_sampleset(
    jm_sampleset: jm.SampleSet,
):
    def f():
        return jm_sampleset

    experiment = jb.Experiment()

    with experiment:
        solver = jb.Solver(f)
        record = solver([])
        experiment.append(record)

    expected_columns = [
        "num_occurrences",
        "energy",
        "objective",
        "onehot1_violations",
        "onehot2_violations",
        "num_samples",
        "num_feasible",
        "execution_time",
    ]

    assert set(expected_columns) <= set(experiment.table.columns)
    assert set(expected_columns) == set(experiment.response_table.columns)
    assert experiment.params_table.empty
