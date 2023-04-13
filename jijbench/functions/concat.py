from __future__ import annotations

import pathlib
import typing as tp

import jijmodeling as jm
import numpy as np
import pandas as pd

from jijbench.consts.default import DEFAULT_RESULT_DIR
from jijbench.elements.base import Any
from jijbench.elements.id import ID
from jijbench.node.base import DataNode, FunctionNode
from jijbench.typing import ConcatableT

if tp.TYPE_CHECKING:
    from typing_extensions import TypeGuard

    from jijbench.containers.containers import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.solver.jijzept import SampleSet


def _is_artifact_list(inputs: list[ConcatableT]) -> TypeGuard[list[Artifact]]:
    from jijbench.containers.containers import Artifact

    return all([isinstance(node, Artifact) for node in inputs])


def _is_experiment_list(inputs: list[ConcatableT]) -> TypeGuard[list[Experiment]]:
    from jijbench.experiment.experiment import Experiment

    return all([isinstance(node, Experiment) for node in inputs])


def _is_record_list(inputs: list[ConcatableT]) -> TypeGuard[list[Record]]:
    from jijbench.containers.containers import Record

    return all([isinstance(node, Record) for node in inputs])


def _is_table_list(inputs: list[ConcatableT]) -> TypeGuard[list[Table]]:
    from jijbench.containers.containers import Table

    return all([isinstance(node, Table) for node in inputs])


def _is_sampleset_list(inputs: list[ConcatableT]) -> TypeGuard[list[SampleSet]]:
    from jijbench.solver.jijzept import SampleSet

    return all([isinstance(node, SampleSet) for node in inputs])


def _is_datanode_list(inputs: list[ConcatableT]) -> bool:
    sample = inputs[0]
    is_datanode = isinstance(sample, DataNode)
    return all([isinstance(node, sample.__class__) for node in inputs]) & is_datanode


class Concat(FunctionNode[ConcatableT, ConcatableT]):
    """Concat class for concatenating multiple data nodes.

    This class can be apply to `Artifact`, `Experiment`, `Record`, `Table`, `SampleSet`.
    """

    @tp.overload
    def __call__(self, inputs: list[Artifact], name: tp.Hashable = None) -> Artifact:
        """Concatenate the gitven multiple 'Artifact' objects.

        Args:
            inputs (list[Artifact]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Artifact: The resulting 'Artifact' object.
        """
        ...

    @tp.overload
    def __call__(
        self,
        inputs: list[Experiment],
        name: str | None = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Experiment:
        """Concatenate the gitven multiple 'Experiment' objects.

        Args:
            inputs (list[Experiment]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.
            autosave (bool, optional): If True, the resulting object will be saved to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory where the resulting object will be saved. Defaults to DEFAULT_RESULT_DIR.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Experiment: The resulting 'Experiment' object.
        """
        ...

    @tp.overload
    def __call__(self, inputs: list[Record], name: tp.Hashable = None) -> Record:
        """Concatenate the gitven multiple 'Record' objects.

        Args:
            inputs (list[Record]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Record: The resulting 'Record' object.
        """
        ...

    @tp.overload
    def __call__(
        self,
        inputs: list[Table],
        name: tp.Hashable = None,
        *,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Table:
        """Concatenate the gitven multiple 'Artifact' objects.

        Args:
            inputs (list[Table]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.
            axis (tp.Literal[0, 1], optional): The axis along which to concatenate the input 'Table' objects. Defaults to 0.
            index_name (str | None, optional): The name of the resulting object's index. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Table: The resulting 'Table' object.
        """
        ...

    @tp.overload
    def __call__(self, inputs: list[SampleSet], name: str) -> SampleSet:
        """Concatenate the gitven multiple 'SampleSet' objects.

        Args:
            inputs (list[SampleSet]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            SampleSet: The resulting 'SampleSet' object.
        """
        ...

    def __call__(
        self,
        inputs: list[tp.Any],
        name: tp.Hashable = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> tp.Any:
        """Concatenate Given a list of either 'Artifact', 'Experiment', 'Record', or 'Table' objects."""
        if _is_datanode_list(inputs):
            return super().__call__(
                inputs,
                name=name,
                autosave=autosave,
                savedir=savedir,
                axis=axis,
                index_name=index_name,
            )
        else:
            raise TypeError(
                "Type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'."
            )

    @tp.overload
    def operate(self, inputs: list[Artifact], name: tp.Hashable = None) -> Artifact:
        """Concatenate the gitven multiple 'Artifact' objects.

        Args:
            inputs (list[Artifact]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Artifact: The resulting 'Artifact' object.
        """
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Experiment],
        name: str | None = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        """Concatenate the gitven multiple 'Experiment' objects.

        Args:
            inputs (list[Experiment]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.
            autosave (bool, optional): If True, the resulting object will be saved to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory where the resulting object will be saved. Defaults to DEFAULT_RESULT_DIR.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Experiment: The resulting 'Experiment' object.
        """
        ...

    @tp.overload
    def operate(self, inputs: list[Record], name: tp.Hashable = None) -> Record:
        """Concatenate the gitven multiple 'Record' objects.

        Args:
            inputs (list[Record]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Record: The resulting 'Record' object.
        """
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Table],
        name: tp.Hashable = None,
        *,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Table:
        """Concatenate the gitven multiple 'Artifact' objects.

        Args:
            inputs (list[Table]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.
            axis (tp.Literal[0, 1], optional): The axis along which to concatenate the input 'Table' objects. Defaults to 0.
            index_name (str | None, optional): The name of the resulting object's index. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            Table: The resulting 'Table' object.
        """
        ...

    @tp.overload
    def operate(self, inputs: list[SampleSet], name: str) -> SampleSet:
        """Concatenate the gitven multiple 'SampleSet' objects.

        Args:
            inputs (list[SampleSet]): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            SampleSet: The resulting 'SampleSet' object.
        """
        ...

    def operate(
        self,
        inputs: list[tp.Any],
        name: tp.Hashable = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> tp.Any:
        """Concatenate Given a list of either 'Artifact', 'Experiment', 'Record', or 'Table' objects."""
        if _is_artifact_list(inputs):
            data = {}
            for node in inputs:
                for k, v in node.data.items():
                    if k in data:
                        data[k].update(v.copy())
                    else:
                        data[k] = v.copy()
            return type(inputs[0])(data, name)
        elif _is_experiment_list(inputs):
            concat_a: Concat[Artifact] = Concat()
            concat_t: Concat[Table] = Concat()
            inputs_a = [n.data[0] for n in inputs]
            inputs_t = [n.data[1] for n in inputs]
            artifact = inputs_a[0].apply(concat_a, inputs_a[1:])
            table = inputs_t[0].apply(
                concat_t, inputs_t[1:], axis=axis, index_name=index_name
            )

            if name is None:
                name = ID().data

            if not isinstance(name, str):
                raise TypeError("Attirbute name of Experiment must be string.")

            return type(inputs[0])(
                (artifact, table),
                name,
                autosave=autosave,
                savedir=savedir,
            )
        elif _is_record_list(inputs):
            data = pd.concat([node.data for node in inputs])
            return type(inputs[0])(data, name)
        elif _is_table_list(inputs):
            data = pd.concat([node.data for node in inputs], axis=axis)
            data = data.fillna(Any(np.nan, "NaN"))
            data.index.name = index_name
            return type(inputs[0])(data, name)
        elif _is_sampleset_list(inputs):
            data = jm.concatenate([node.data for node in inputs])
            return type(inputs[0])(data, str(name))
        else:
            raise TypeError(
                "Type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'."
            )
