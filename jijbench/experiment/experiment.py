from __future__ import annotations

import abc
import datetime
import pathlib
import typing as tp
import uuid
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jijbench.consts.default import DEFAULT_RESULT_DIR
from jijbench.containers.containers import Artifact, Container, Record, Table
from jijbench.elements.id import ID
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.io.io import save
from jijbench.solver.base import Parameter, Response
from jijbench.typing import ArtifactDataType, ExperimentDataType


@dataclass
class Experiment(Container[ExperimentDataType]):
    """Stores data related to an benchmark.

    The Experiment class stores the results obtained from a benchmark as Artifact and Table objects and assists in managing the benchmark process.
    With this class, you can add and save experimental results, as well as view them in various formats.

    Attributes:
        data (tuple[Artifact, Table]): A tuple containing an Artifact object and a Table object.
        name (str): The name of the experiment.
        autosave (bool): Whether to automatically save the experiment upon exit.
        savedir (str | pathlib.Path): The directory where the experiment will be saved.
    """

    data: tuple[Artifact, Table] = field(default_factory=lambda: (Artifact(), Table()))
    name: str = field(default_factory=lambda: str(uuid.uuid4()))
    autosave: bool = field(default=True, repr=False)
    savedir: str | pathlib.Path = field(default=DEFAULT_RESULT_DIR, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.savedir = pathlib.Path(self.savedir)
        setattr(self, "state", _Created())

    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        a_len = len(self.data[0])
        t_len = len(self.data[1])
        if a_len != t_len:
            warnings.warn(
                f"The length of artifact object and table object are different: {a_len} != {t_len}, return the larger one."
            )
        return max(a_len, t_len)

    def __enter__(self) -> Experiment:
        """Set up Experiment.
        Automatically makes a directory for saving the experiment, if it doesn't exist.
        """

        setattr(self, "state", _Running(self.name))
        pathlib.Path(self.savedir).mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Saves the experiment if autosave is True."""
        state = getattr(self, "state")

        if self.autosave:
            state.save(self)

        setattr(self, "state", _Waiting(self.name))

    @property
    def artifact(self) -> ArtifactDataType:
        """Return the artifact of the experiment as a dictionary."""
        return self.data[0].view()

    @property
    def table(self) -> pd.DataFrame:
        """Return the table of the experiment as a pandas dataframe."""
        return self.data[1].view()

    @property
    def params_table(self) -> pd.DataFrame:
        """Return the parameters table of the experiment as a pandas dataframe."""
        _, t = self.data
        bools = t.data.apply(lambda x: not isinstance(x[0], Parameter))
        droped_columns = t.data.columns[bools]
        data = t.data.drop(columns=droped_columns)
        return Table(data, self.name).view()

    @property
    def response_table(self) -> pd.DataFrame:
        """Return the returns table of the experiment as a pandas dataframe."""
        _, t = self.data
        bools = t.data.apply(lambda x: not isinstance(x[0], Response))
        droped_columns = t.data.columns[bools]
        data = t.data.drop(columns=droped_columns)
        return Table(data, self.name).view()

    @classmethod
    def validate_data(cls, data: ExperimentDataType) -> ExperimentDataType:
        """Validate the data of the experiment.

        Args:
            data (ExperimentDataType): The data to validate.

        Raises:
            TypeError: If data is not an instance of ExperimentDataType or
            if the first element of data is not an instance of Artifact or
            if the second element of data is not an instance of Table.

        Returns:
            ExperimentDataType: The validated data.
        """
        if not isinstance(data, tuple):
            raise TypeError(f"Data must be a tuple, got {type(data)}.")

        if len(data) != 2:
            raise ValueError(f"Data must be a tuple of length 2, got {len(data)}.")

        artifact, table = data
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[0] must be Artifact instead of {type(artifact).__name__}."
            )
        if not isinstance(table, Table):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[1] must be Table instead of {type(artifact).__name__}."
            )
        return data

    def view(self) -> tuple[dict, pd.DataFrame]:
        """Return a tuple of the artifact dictionary and table dataframe."""
        return (self.data[0].view(), self.data[1].view())

    def append(self, record: Record) -> None:
        """Append a new record to the experiment.

        Args:
            record (Record): The record to be appended to the experiment.
        """
        state = getattr(self, "state")
        state.append(self, record)

    def star(self) -> None:
        """Mark the experiment."""

        savedir = pathlib.Path(self.savedir)
        benchmark_id = None
        if (savedir / "star.csv").exists():
            star_file = savedir / "star.csv"
        elif (savedir.parent / "star.csv").exists():
            star_file = savedir.parent / "star.csv"
            benchmark_id = savedir.name
        else:
            star_file = savedir / "star.csv"

        if star_file.exists():
            if star_file.stat().st_size:
                star = pd.read_csv(star_file, index_col=0)
            else:
                star = pd.DataFrame(
                    columns=["benchmark_id", "experiment_id", "savedir"]
                )
            star.loc[len(star), ["benchmark_id", "experiment_id", "savedir"]] = [
                benchmark_id or np.nan,
                self.name,
                self.savedir,
            ]
            star = star.drop_duplicates(subset=["experiment_id"])
        else:
            star_file.parent.mkdir(parents=True, exist_ok=True)
            star = pd.DataFrame(
                {
                    "benchmark_id": benchmark_id or np.nan,
                    "experiment_id": [self.name],
                    "savedir": [self.savedir],
                }
            )
        star.to_csv(star_file)

    def save(self):
        """Save the experiment."""
        pathlib.Path(self.savedir).mkdir(parents=True, exist_ok=True)
        state = getattr(self, "state")
        state.save(self)


class _ExperimentState(metaclass=abc.ABCMeta):
    def __init__(self, index: tp.Hashable = 0) -> None:
        self.index = index

    @abc.abstractmethod
    def append(self, context: Experiment, record: Record) -> None:
        pass

    @abc.abstractmethod
    def save(self, context: Experiment) -> None:
        pass


class _Created(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        record.name = self.index
        _append(context, record)
        if isinstance(self.index, int):
            self.index += 1
        context.state = _Waiting(self.index)

    def save(self, context: Experiment) -> None:
        save(context, savedir=context.savedir)


class _Waiting(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        record.name = self.index
        _append(context, record)
        if isinstance(self.index, int):
            self.index += 1

    def save(self, context: Experiment) -> None:
        save(context, savedir=context.savedir)


class _Running(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        run_id = ID().data
        if isinstance(self.index, str):
            record.name = (self.index, run_id)
        elif isinstance(self.index, tp.Iterable):
            record.name = (*self.index, run_id)
        else:
            record.name = (self.index, run_id)
        _append(context, record)

    def save(
        self,
        context: Experiment,
    ) -> None:
        a, t = context.data
        latest = t.index[-1]

        ai = Artifact({latest: a.data[latest]}, a.name)
        ti = Table(t.data.loc[[latest]], t.name)

        experiment = Experiment(
            (ai, ti), context.name, autosave=context.autosave, savedir=context.savedir
        )
        save(experiment, savedir=context.savedir, mode="a")


def _append(context: Experiment, record: Record) -> None:
    concat: Concat[Experiment] = Concat()
    data = (ArtifactFactory()([record]), TableFactory()([record]))
    other = type(context)(
        data, context.name, autosave=context.autosave, savedir=context.savedir
    )
    node = context.apply(
        concat,
        [other],
        name=context.name,
        autosave=context.autosave,
        savedir=context.savedir,
    )
    context._update_attrs(node)


def get_benchmark_ids(
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
) -> list[str | float]:
    """Get the benchmark ids.

    Args:
        savedir (str | pathlib.Path, optional): The directory to save the experiment. Defaults to DEFAULT_RESULT_DIR.

    Returns:
        list[str]: The benchmark ids.
    """

    benchmark_ids: list[str | float] = []
    for d in pathlib.Path(savedir).glob("*"):
        if d.is_dir():
            if list(d.glob("artifact")):
                benchmark_ids.append(np.nan)
            else:
                benchmark_ids.extend([d.name] * len(get_experiment_ids(d)))

    return benchmark_ids


def get_experiment_ids(
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR, only_star: bool = False
) -> list[str]:
    """Get the experiment ids.

    Args:
        savedir (str | pathlib.Path, optional): The directory to save the experiment. Defaults to DEFAULT_RESULT_DIR.
        only_star (bool, optional): Whether to only return the starred experiments. Defaults to False.

    Returns:
        list[str]: The experiment ids.
    """
    if only_star:
        star_file = pathlib.Path(savedir) / "star.csv"
        if star_file.exists():
            return pd.read_csv(star_file, index_col=0)["experiment_id"].tolist()
        else:
            return []
    else:
        experiment_ids: list[str] = []
        for d in pathlib.Path(savedir).glob("*"):
            if d.is_dir():
                if list(d.glob("artifact")):
                    experiment_ids.append(d.name)
                else:
                    experiment_ids.extend(get_experiment_ids(d))
        return experiment_ids


def get_id_table(
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
) -> pd.DataFrame:
    """Get the experiment info.

    Args:
        savedir (str | pathlib.Path, optional): The directory to save the experiment. Defaults to DEFAULT_RESULT_DIR.

    Returns:
        pd.DataFrame: The experiment info.
    """

    info = pd.DataFrame(
        {
            "benchmark_id": get_benchmark_ids(savedir),
            "experiment_id": get_experiment_ids(savedir),
        }
    )

    info["savedir"] = info.apply(
        lambda x: f"{savedir}/{x['benchmark_id']}/{x['experiment_id']}"
        if isinstance(x["benchmark_id"], str)
        else f"{savedir}/{x['experiment_id']}",
        axis=1,
    )

    info["timestamp"] = info["savedir"].apply(
        lambda x: datetime.datetime.fromtimestamp(
            pathlib.Path(x).stat().st_ctime
        ).strftime("%Y-%m-%d %H:%M:%S")
    )
    info["star"] = ""

    star_file = pathlib.Path(savedir) / "star.csv"
    if star_file.exists():
        star = pd.read_csv(star_file, index_col=0)
        mask = info["experiment_id"].isin(star["experiment_id"].tolist())
        info.loc[mask, "star"] = "*"
    return info
