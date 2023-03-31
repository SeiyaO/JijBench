from __future__ import annotations

import typing as tp

import numpy as np

from jijbench.elements.base import Number
from jijbench.node.base import FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.elements.array import Array


class Min(FunctionNode["Array", Number]):
    """Calculate the minimum value of an input `Array`.

    The `Min` class is a subclass of `FunctionNode` that calculates the minimum value of an input `Array`
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculate the minimum value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.min, "min")


class Max(FunctionNode["Array", Number]):
    """Calculate the maximum value of an input `Array`.

    The `Max` class is a subclass of `FunctionNode` that calculates the maximum value of an input `Array`
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculate the maximum value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.max, "max")


class Mean(FunctionNode["Array", Number]):
    """Calculate the mean value of an input `Array`.

    The `Mean` class is a subclass of `FunctionNode` that calculates the mean value of an input `Array`
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculate the mean value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.mean)


class Std(FunctionNode["Array", Number]):
    """Calculate the standard deviation of an input `Array`.

    The `Std` class is a subclass of `FunctionNode` that calculates the standard deviation of an input `Array`
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculate the standard deviation of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.std)


def _operate_array(
    inputs: list[Array], f: tp.Callable, f_name: str | None = None
) -> Number:
    data = f(inputs[0].data)
    if "int" in str(data.dtype):
        data = int(data)
    else:
        data = float(data)

    if f_name is None:
        f_name = f.__name__
    name = f"{inputs[0].name}_{f_name}"
    return Number(data, name)
