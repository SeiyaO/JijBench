from __future__ import annotations

import pathlib
import typing as tp
from unittest.mock import MagicMock

import jijmodeling as jm
import jijzept as jz
import pytest
from _pytest.fixtures import SubRequest
from pytest_mock import MockerFixture

import jijbench as jb


@pytest.fixture
def onehot_problem() -> jm.Problem:
    d = jm.Placeholder("d", ndim=1)
    x = jm.BinaryVar("x", shape=(d.shape[0].set_latex("n")))
    i = jm.Element("i", belong_to=d.shape[0])
    problem = jm.Problem("problem")
    problem += jm.sum(i, d[i] * x[i])
    problem += jm.Constraint("onehot1", jm.sum(i, x[i]) == 1)
    return problem


@pytest.fixture
def knapsack_problem() -> jm.Problem:
    return jb.get_problem("knapsack")


@pytest.fixture
def tsp_problem() -> jm.Problem:
    return jb.get_problem("travelling-salesman")


@pytest.fixture
def knapsack_instance_data() -> jm.PH_VALUES_INTERFACE:
    return jb.get_instance_data("knapsack")[0]


@pytest.fixture
def tsp_instance_data() -> jm.PH_VALUES_INTERFACE:
    return jb.get_instance_data("travelling-salesman")[0]


@pytest.fixture
def jm_sampleset_dict() -> dict[str, tp.Any]:
    return {
        "record": {
            "solution": {
                "x": [
                    (([0, 1], [0, 1]), [1, 1], (2, 2)),
                    (([0, 1], [1, 0]), [1, 1], (2, 2)),
                    (([], []), [], (2, 2)),
                    (([0, 1], [0, 0]), [1, 1], (2, 2)),
                ]
            },
            "num_occurrences": [4, 3, 2, 1],
        },
        "evaluation": {
            "energy": [3.0, 24.0, 0.0, 20.0],
            "objective": [3.0, 24.0, 0.0, 17.0],
            "constraint_violations": {
                "onehot1": [0.0, 0.0, 2.0, 0.0],
                "onehot2": [0.0, 0.0, 2.0, 2.0],
            },
            "penalty": {},
        },
        "measuring_time": {"solve": None, "system": None, "total": None},
    }


@pytest.fixture
def jm_sampleset(jm_sampleset_dict: dict) -> jm.SampleSet:
    sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict)
    solving_time = jm.SolvingTime(
        **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
    )
    sampleset.measuring_time.solve = solving_time
    return sampleset


@pytest.fixture
def jm_sampleset_dict_no_constraint() -> dict[str, tp.Any]:
    return {
        "record": {
            "solution": {
                "x": [
                    (([0, 1], [0, 1]), [1, 1], (2, 2)),
                    (([0, 1], [1, 0]), [1, 1], (2, 2)),
                    (([], []), [], (2, 2)),
                    (([0, 1], [0, 0]), [1, 1], (2, 2)),
                ]
            },
            "num_occurrences": [4, 3, 2, 1],
        },
        "evaluation": {
            "energy": [3.0, 24.0, 0.0, 20.0],
            "objective": [3.0, 24.0, 0.0, 17.0],
            "constraint_violations": {},
            "penalty": {},
        },
        "measuring_time": {"solve": None, "system": None, "total": None},
    }


@pytest.fixture
def jm_sampleset_no_constraint(jm_sampleset_dict_no_constraint: dict) -> jm.SampleSet:
    sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict_no_constraint)
    solving_time = jm.SolvingTime(
        **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
    )
    sampleset.measuring_time.solve = solving_time
    return sampleset


@pytest.fixture
def jm_sampleset_dict_no_obj_no_constraint() -> dict[str, tp.Any]:
    return {
        "record": {
            "solution": {
                "x": [
                    (([0, 1], [0, 1]), [1, 1], (2, 2)),
                    (([0, 1], [1, 0]), [1, 1], (2, 2)),
                    (([], []), [], (2, 2)),
                    (([0, 1], [0, 0]), [1, 1], (2, 2)),
                ]
            },
            "num_occurrences": [4, 3, 2, 1],
        },
        "evaluation": {
            "energy": [3.0, 24.0, 0.0, 20.0],
            "constraint_violations": {},
            "penalty": {},
        },
        "measuring_time": {"solve": None, "system": None, "total": None},
    }


@pytest.fixture
def jm_sampleset_no_obj_no_constraint(
    jm_sampleset_dict_no_obj_no_constraint: dict,
) -> jm.SampleSet:
    sampleset = jm.SampleSet.from_serializable(jm_sampleset_dict_no_obj_no_constraint)
    solving_time = jm.SolvingTime(
        **{"preprocess": 1.0, "solve": 1.0, "postprocess": 1.0}
    )
    sampleset.measuring_time.solve = solving_time
    return sampleset


@pytest.fixture
def sa_sampler() -> jz.JijSASampler:
    config_path = pathlib.Path(__file__).parent / "config.toml"
    sampler = jz.JijSASampler(config=config_path)
    return sampler


@pytest.fixture
def sample_model(mocker: MockerFixture, jm_sampleset: jm.SampleSet) -> MagicMock:
    mocker.patch(
        "jijbench.Experiment.save",
    )
    sample_model = mocker.patch(
        "jijzept.JijSASampler.sample_model", autospec=True, return_value=jm_sampleset
    )
    sample_model.__name__ = "sample_model"
    return sample_model
