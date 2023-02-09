from __future__ import annotations

import typing as tp
import inspect

from dataclasses import dataclass
from jijbench.elements.base import Element
from jijbench.exceptions.exceptions import SolverFailedError
from jijbench.node.base import FunctionNode
from jijbench.mappings.mappings import Record
from jijbench.functions.factory import RecordFactory
from jijbench.typing import T

import jijzept as jz

@dataclass
class Parameter(Element[T]):
    """A parameter for a solver function.

    Attributes:
        data (Any): The data in the node.
        name (str): The name of the parameter.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data


@dataclass
class Return(Element[T]):
    """A return value of a solver function.

    Attributes:
        data (Any): The data in the node.
        name (str): The name of the return value.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data


class Solver(FunctionNode[Parameter, Record]):
    """A solver function that takes a list of Parameter and returns a Record.

    Attributes:
        name (str): The name of the solver function.
        function (Callable): The actual function to be executed.
    """

    def __init__(self, function: tp.Callable, name: str | None = None) -> None:
        """The constructor of the `Solver` class.

        Args:
            function (Callable): The actual function to be executed.
            name (str, optional): The name of the solver function. Defaults to None.
        """
        if name is None:
            name = function.__name__
        super().__init__(name)
        self.function = function

    def operate(
        self,
        inputs: list[Parameter],
        is_parsed_sampleset: bool = True,
    ) -> Record:
        """The main operation of the solver function.

        Args:
            inputs (list[Parameter]): The list of input `Parameter` for the solver function.
            is_parsed_sampleset (bool, optional): Whether the sample set is parsed. Defaults to True.

        Raises:
            SolverFailedError: If an error occurs inside the solver function.

        Returns:
            Record: The result of the solver function as a `Record`.
        """
        parameters = inspect.signature(self.function).parameters
        kwargs = (
            kwargs
            if "kwargs" in parameters
            else {k: v for k, v in kwargs.items() if k in parameters}
        )
        return self.function(**kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def ret_names(self):
        return self._ret_names

    @ret_names.setter
    def ret_names(self, names):
        self._ret_names = names

    def to_named_ret(self, ret):
        if isinstance(ret, tuple):
            names = (
                self._ret_names
                if self._ret_names
                else [f"solver_return_values[{i}]" for i in range(len(ret))]
            )
            ret = dict(zip(names, ret))
        else:
            names = self.ret_names if self._ret_names else ["solver_return_values[0]"]
            ret = dict(zip(names, [ret]))
        return ret

    def _parse_solver(self, solver):
        if isinstance(solver, str):
            return getattr(DefaultSolver(), solver)
        elif isinstance(solver, Callable):
            return solver
        else:
            return


class DefaultSolver:
    jijzept_config: Optional[str] = None
    dwave_config: Optional[str] = None
    jijzept_sampler_names = ["JijSASampler", "JijSQASampler", "JijSwapMovingSampler"]

    @property
    def JijSASampler(self):
        return self.jijzept_sa_sampler_sample_model
    
    @property
    def JijSQASampler(self):
        return self.jijzept_sqa_sampler_sample_model
    
    @property
    def JijSwapMovingSampler(self):
        return self.jijzept_swapmoving_sampler_sample_model

    @classmethod
    def jijzept_sa_sampler_sample_model(cls, problem, instance_data, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSASampler,
            problem,
            ph_value,
            **kwargs,
        )
        
    @classmethod
    def jijzept_sqa_sampler_sample_model(cls, problem, ph_value, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSQASampler,
            problem,
            ph_value,
            **kwargs
        )
        
    @classmethod
    def jijzept_swapmoving_sampler_sample_model(cls, problem, ph_value, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSwapMovingSampler,
            problem,
            ph_value,
            **kwargs
        )

    @staticmethod
    def _sample_by_jijzept(sampler, problem, instance_data, sync=True, **kwargs):
        sampler = sampler(config=DefaultSolver.jijzept_config)
        if sync:
            parameters = inspect.signature(sampler.sample_model).parameters
            kwargs = {k: w for k, w in kwargs.items() if k in parameters}
            response = sampler.sample_model(problem, ph_value, sync=sync, **kwargs)
        else:
            response = sampler.get_result(solution_id=kwargs["solution_id"])
        return response
