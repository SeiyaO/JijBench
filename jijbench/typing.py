from __future__ import annotations

import datetime
import typing as tp

import jijmodeling as jm
import pandas as pd

if tp.TYPE_CHECKING:
    from jijbench.containers.containers import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.node.base import DataNode
    from jijbench.solver.jijzept import SampleSet
    from typing_extensions import TypeAlias


# node
T = tp.TypeVar("T")
DataNodeInT = tp.TypeVar("DataNodeInT", bound="DataNode[tp.Any]")
DataNodeOutT = tp.TypeVar("DataNodeOutT", bound="DataNode[tp.Any]")
ConcatableT = tp.TypeVar(
    "ConcatableT", "Artifact", "Experiment", "Record", "SampleSet", "Table"
)


# element
DateTypes: TypeAlias = tp.Union[str, datetime.datetime, pd.Timestamp]
NumberTypes: TypeAlias = tp.Union[int, float]

# solver
ModelType: TypeAlias = tp.Tuple[jm.Problem, jm.PH_VALUES_INTERFACE]


# containers
ArtifactKeyType: TypeAlias = tp.Hashable
ArtifactValueType: TypeAlias = tp.Dict[tp.Hashable, "DataNode[tp.Any]"]
ArtifactDataType: TypeAlias = tp.Dict[ArtifactKeyType, ArtifactValueType]
ExperimentDataType: TypeAlias = tp.Tuple["Artifact", "Table"]
