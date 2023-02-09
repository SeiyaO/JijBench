from __future__ import annotations

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
        return _operate_array(inputs, np.min)


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
        return _operate_array(inputs, np.max)


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


class Std(FunctionNode[Array, Array]):
    def operate(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}"
        node = Array(data, name)
        return node
