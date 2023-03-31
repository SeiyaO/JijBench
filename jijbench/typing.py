from __future__ import annotations

import datetime
import typing as tp

import jijmodeling as jm
import pandas as pd
from typing_extensions import TypeAlias

if tp.TYPE_CHECKING:
    from jijbench.containers.containers import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.node.base import DataNode

T = tp.TypeVar("T")


# node
DataNodeT = tp.TypeVar("DataNodeT", bound="DataNode[tp.Any]")
DataNodeInT = tp.TypeVar("DataNodeInT", bound="DataNode[tp.Any]")
DataNodeOutT = tp.TypeVar("DataNodeOutT", bound="DataNode[tp.Any]")


# element
DateTypes: TypeAlias = tp.Union[str, datetime.datetime, pd.Timestamp]
NumberTypes: TypeAlias = tp.Union[int, float]

# solver
ModelType: TypeAlias = tp.Tuple[jm.Problem, jm.PH_VALUES_INTERFACE]


# mapping
ContainerT = tp.TypeVar("ContainerT", "Artifact", "Experiment", "Record", "Table")
ContainerTypes: TypeAlias = tp.Union["Artifact", "Experiment", "Record", "Table"]
ContainerListTypes: TypeAlias = tp.Union[
    tp.List["Artifact"], tp.List["Experiment"], tp.List["Record"], tp.List["Table"]
]
ArtifactDataType: TypeAlias = tp.Dict[
    tp.Hashable, tp.Dict[tp.Hashable, "DataNode[tp.Any]"]
]
ExperimentDataType: TypeAlias = tp.Tuple["Artifact", "Table"]
