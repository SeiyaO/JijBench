from __future__ import annotations

import abc
import typing as tp

import pandas as pd

from jijbench.node.base import FunctionNode
from jijbench.typing import DataNodeInT, DataNodeOutT

if tp.TYPE_CHECKING:
    from jijbench.containers.containers import Artifact, Record, Table


class Factory(FunctionNode[DataNodeInT, DataNodeOutT]):
    """An abstract base class for creating a new data node from a list of input nodes.
    
    Attributes:
        inputs (list[`DataNodeInT`]): List of input data nodes.
        name (Optional[str]): Name of the resulting data node.
    """

    @abc.abstractmethod
    def create(
        self, inputs: list[DataNodeInT], name: str | None = None
    ) -> DataNodeOutT:
        """Abstract method to create a new data node.
        Subclasses must implement this method.

        Args:
            inputs (List[DataNodeInT]): List of input data nodes.
            name (str | None): Name of the resulting data node.

        Returns:
            DataNodeOutT: The resulting data node.

        """
        pass

    def operate(
        self, inputs: list[DataNodeInT], name: str | None = None, **kwargs: tp.Any
    ) -> DataNodeOutT:
        """Create a new data node from the given inputs.
        This method calls `create` method to create a new data node from the given inputs.

        Args:
            inputs (list[DataNodeInT]): List of input data nodes.
            name (str | None, optional): Name of the resulting data node.
            **kwargs: Additional keyword arguments.

        Returns:
            DataNodeOutT: The resulting data node.

        """
        return self.create(inputs, name, **kwargs)


class RecordFactory(Factory[DataNodeInT, "Record"]):
    """A factory class for creating Record objects.

    This class creates Record objects from a list of input DataNode objects. It uses the `create` method to
    process the input DataNodes, extract their data and convert it into a pandas Series. The resulting Series is
    used to create the Record object. The class also includes a helper method `_to_nodes_from_sampleset` which
    is used to extract the relevant data from `jijmodeling.SampleSet` objects.
    """

    def create(
        self,
        inputs: list[DataNodeInT],
        name: str = "",
    ) -> Record:
        """Create a Record object from the input DataNode objects.

        This method takes a list of input DataNode objects, processes them and converts them into a pandas
        Series. The resulting Series is used to create the Record object.

        Args:
            inputs (list[DataNodeInT]): A list of DataNode objects to be processed.
            name (str, optional): A name for the Record object. Defaults to "".

        Returns:
            Record: A Record object created from the processed input DataNode objects.
        """
        from jijbench.containers.containers import Record

        data = pd.Series({node.name: node for node in inputs})
        return Record(data, name)


class ArtifactFactory(Factory["Record", "Artifact"]):
    """A factory class for creating Artifact objects."""

    def create(self, inputs: list[Record], name: str = "") -> Artifact:
        """Creates an `Artifact` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to create the `Artifact`.
            name (str, optional): Name of the `Artifact` object. Defaults to an empty string.

        Returns:
            Artifact: The created `Artifact` object.
        """
        from jijbench.containers.containers import Artifact

        data = {
            node.name
            if isinstance(node.name, tp.Hashable)
            else str(node.name): node.data.to_dict()
            for node in inputs
        }
        return Artifact(data, name)


class TableFactory(Factory["Record", "Table"]):
    """A factory class for creating Table objects."""

    def create(
        self,
        inputs: list[Record],
        name: str = "",
        index_name: str | None = None,
    ) -> Table:
        """Creates a `Table` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to create the `Table`.
            name (str, optional): Name of the `Table` object. Defaults to an empty string.
            index_name (str | None, optional): Name of the index in the created `Table`. Defaults to None.

        Returns:
            Table: The created `Table` object.
        """
        from jijbench.containers.containers import Table

        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return Table(data, name)
