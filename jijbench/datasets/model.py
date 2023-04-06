from __future__ import annotations

import glob
import json
import pathlib
import typing as tp

import jijmodeling as jm

from jijbench.datasets.problems import (
    bin_packing,
    knapsack,
    nurse_scheduling,
    travelling_salesman,
    travelling_salesman_with_time_windows,
)


class InstanceDataFileStorage:
    base_dir = pathlib.Path(__file__).parent.joinpath("Instances")

    def __init__(
        self,
        problem_name: tp.Literal[
            "bin-packing",
            "knapsack",
            "nurse-scheduling",
            "travelling-salesman",
            "travelling-salesman-with-time-windows",
        ],
    ):
        self.problem_name = problem_name

    def get_names(
        self,
        size: tp.Literal["small", "medium", "large"] = "small",
        n: int | tp.Literal["all"] = 5,
    ) -> list[str]:
        return list(map(lambda x: pathlib.Path(x).name, self.get_files(size, n)))

    def get_files(
        self,
        size: tp.Literal["small", "medium", "large"] = "small",
        n: int | tp.Literal["all"] = 5,
    ) -> list[str]:
        files = glob.glob(
            f"{self.base_dir}/{size}/{self.problem_name}/**/*.json", recursive=True
        )
        files.sort()
        if n == "all":
            return files
        elif isinstance(n, int):
            return files[:n]
        else:
            raise TypeError(f"n must be 'all' or int, but {type(n)} is given.")

    def get_files_map(
        self,
        size: tp.Literal["small", "medium", "large"] = "small",
        n: int | tp.Literal["all"] = 5,
    ) -> dict[str, str]:
        return {pathlib.Path(file).name: file for file in self.get_files(size, n)}

    @classmethod
    def load(cls, file: str) -> jm.PH_VALUES_INTERFACE:
        if not pathlib.Path(file).exists():
            raise FileNotFoundError(f"'{file}' is not found.")

        with open(file, "r") as f:
            instance_data = json.load(f)
        return instance_data

    def load_instance_data(
        self,
        size: tp.Literal["small", "medium", "large"] = "small",
        n: int | tp.Literal["all"] = 5,
    ) -> list[jm.PH_VALUES_INTERFACE]:
        return [self.load(file) for file in self.get_files(size, n)]


def get_models(
    problem_name: tp.Literal[
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ],
    size: tp.Literal["small", "medium", "large"] = "small",
    n: int | tp.Literal["all"] = 5,
) -> list[tuple[jm.Problem, jm.InstanceData]]:
    storage = InstanceDataFileStorage(problem_name)
    problem = get_problem(problem_name)
    files = storage.load_instance_data(size, n)
    return list(zip([problem] * n, files))


def get_problem(
    problem_name: tp.Literal[
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ]
) -> jm.Problem:
    if problem_name == "bin-packing":
        return bin_packing()
    elif problem_name == "knapsack":
        return knapsack()
    elif problem_name == "nurse-scheduling":
        return nurse_scheduling()
    elif problem_name == "travelling-salesman":
        return travelling_salesman()
    elif problem_name == "travelling-salesman-with-time-windows":
        return travelling_salesman_with_time_windows()
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")


def get_instance_data(
    problem_name: tp.Literal[
        "bin-packing",
        "knapsack",
        "nurse-scheduling",
        "travelling-salesman",
        "travelling-salesman-with-time-windows",
    ],
    size: tp.Literal["small", "medium", "large"] = "small",
    n: int | tp.Literal["all"] = 5,
) -> list[jm.PH_VALUES_INTERFACE]:
    storage = InstanceDataFileStorage(problem_name)
    return storage.load_instance_data(size, n)
