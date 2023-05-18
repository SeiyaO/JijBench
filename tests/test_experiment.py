from __future__ import annotations

import pathlib
import shutil
import typing as tp

import jijmodeling as jm
import numpy as np
import pandas as pd
import pytest

import jijbench as jb
from jijbench.consts.default import DEFAULT_RESULT_DIR
from jijbench.experiment.experiment import _Created, _Running, _Waiting


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    if DEFAULT_RESULT_DIR.exists():
        shutil.rmtree(DEFAULT_RESULT_DIR)


def f1():
    return "Hello, World!"


def f2(i: int) -> int:
    return i**2


def f3(i: int) -> tuple[int, int]:
    return i + 1, i + 2


def f4(i: int, j: int = 1) -> int:
    return i + j


def simple_experiment(name: str, savedir: pathlib.Path) -> jb.Experiment:
    experiment = jb.Experiment(name=name, savedir=savedir)

    solver = jb.Solver(f1)
    record = solver([])
    experiment.append(record)

    return experiment


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

        actual = experiment.table.filter(regex="solver_output").tail(1)
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

            actual = experiment.table.filter(regex="solver_output").tail(1)
            expected = pd.DataFrame([f(*v)], index=actual.index, columns=actual.columns)

            assert actual.index.get_level_values(0)[-1] == experiment.name
            assert actual.equals(expected)

            state = getattr(experiment, "state")
            assert isinstance(state, _Running)
    experiment.save()


def test_experiment_with_solver_outputing_jm_sampleset(
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
        "total_violations",
        "num_samples",
        "num_feasible",
        "execution_time",
    ]

    assert set(expected_columns) <= set(experiment.table.columns)
    assert set(expected_columns) == set(experiment.response_table.columns)
    assert experiment.params_table.empty


def test_star():
    savedir = DEFAULT_RESULT_DIR
    name = "star_experiment"
    experiment = simple_experiment(name, savedir)
    experiment.star()

    star_file = savedir / "star.csv"
    assert star_file.exists()

    star = pd.read_csv(star_file, index_col=0)
    assert star.isin([np.nan, experiment.name, str(experiment.savedir)]).any().all()


def test_get_benchmark_ids():
    savedir = DEFAULT_RESULT_DIR
    experiment = simple_experiment(name="example", savedir=savedir)
    experiment.save()

    benchmark_ids = jb.get_benchmark_ids()

    assert np.isnan(benchmark_ids).all()


@pytest.mark.parametrize("only_star", [True, False])
def test_get_experiment_ids(only_star: bool):
    savedir = DEFAULT_RESULT_DIR
    experiment = simple_experiment(name="example", savedir=savedir)
    star_experiment = simple_experiment(name="example_star", savedir=savedir)

    experiment_ids = jb.get_experiment_ids(only_star=only_star)
    assert not experiment_ids

    experiment.save()

    star_experiment.star()
    star_experiment.save()

    experiment_ids = jb.get_experiment_ids(only_star=only_star)
    if only_star:
        assert len(experiment_ids) == 1
        assert experiment_ids[0] == star_experiment.name
    else:
        assert set(experiment_ids) == {experiment.name, star_experiment.name}
