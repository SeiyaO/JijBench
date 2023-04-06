from __future__ import annotations

import datetime
import typing as tp

import jijmodeling as jm
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if tp.TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from jijbench.containers.containers import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.node.base import DataNode
    from jijbench.solver.jijzept import SampleSet


# node
T = tp.TypeVar("T")
DataNodeInT = tp.TypeVar("DataNodeInT", bound="DataNode[tp.Any]")
DataNodeOutT = tp.TypeVar("DataNodeOutT", bound="DataNode[tp.Any]")
ConcatableT = tp.TypeVar(
    "ConcatableT", "Artifact", "Experiment", "Record", "SampleSet", "Table"
)


# element
ArrayType: TypeAlias = NDArray[tp.Union[np.int64, np.float64]]
DateTypes: TypeAlias = tp.Union[str, datetime.datetime, pd.Timestamp]
NumberTypes: TypeAlias = tp.Union[int, float, np.int64, np.float64]

# solver
ModelType: TypeAlias = tp.Tuple[jm.Problem, jm.PH_VALUES_INTERFACE]


# containers
ArtifactKeyType: TypeAlias = tp.Hashable
ArtifactValueType: TypeAlias = tp.Dict[tp.Hashable, "DataNode[tp.Any]"]
ArtifactDataType: TypeAlias = tp.Dict[ArtifactKeyType, ArtifactValueType]
ExperimentDataType: TypeAlias = tp.Tuple["Artifact", "Table"]
