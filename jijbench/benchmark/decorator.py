from __future__ import annotations

import inspect
import pathlib
import typing as tp
from functools import wraps

import pandas as pd

import jijbench as jb
from jijbench.consts.path import DEFAULT_RESULT_DIR


def checkpoint(
    name: str | None = None, savedir: str | pathlib.Path = DEFAULT_RESULT_DIR
):
    """Decorator for saving and checkpointing the results of a function.

    Args:
        name (str, optional): The name of the benchmark. Defaults to None.
        savedir (str | pathlib.Path, optional): The directory where the benchmark will be saved. Defaults to DEFAULT_RESULT_DIR.

    Returns:
        Callable: The decorated function.

    Examples:
        ```python
        # case POSITIONAL OR KEYWORD
        @checkpoint(name="example_checkpoint")
        def f1(cha: str, i: int = 1) -> str:
            return f"{cha}-{i}"

        # case KEYWORD_ONLY
        @checkpoint(name="example_checkpoint")
        def f2(cha: str, *, f: float=1.0) -> str:
            return f"{cha}-{f}"

        # case VAR_KEYWORD
        # Variable positional arguments like *args are not supported.
        @checkpoint(name="example_checkpoint")
        def f3(cha: str, **kwargs) -> str:
            return f"{cha}-{kwargs}"

        f1("1", 2)
        f2("1", f=2.0)
        f3("1", f=2.0, g=3.0)

        bench = jb.load("example_checkpoint")
        ```
    """

    def decorator(func: tp.Callable[..., tp.Any]):
        @wraps(func)
        def wrapper(*args: tp.Any, **kwargs: tp.Any):
            params = {}
            signature = inspect.signature(func)
            pos_arg_index = 0
            kw_arg_index = 0
            # Make benchmark parameters by inspecting the function signature.
            # CASE 1: POSITIONAL OR KEYWORD is v.kind = 1
            for k, v in signature.parameters.items():
                if v.kind == 1:
                    if pos_arg_index < len(args):
                        params[k] = [args[pos_arg_index]]
                        pos_arg_index += 1
                elif v.kind == 2:
                    raise TypeError("Variable positional arguments are not supported.")
                elif v.kind == 3:
                    if k in kwargs:
                        params[k] = [kwargs[k]]
                        kw_arg_index += 1
                elif v.kind == 4:
                    while kw_arg_index < len(kwargs):
                        k = list(kwargs.keys())[kw_arg_index]
                        if k in kwargs:
                            params[k] = [kwargs[k]]
                            kw_arg_index += 1
            bench = jb.Benchmark(params=params, solver=func, name=name)
            if name:
                bench_dir = pathlib.Path(savedir) / name
                if bench_dir.exists():
                    jb.load(bench_dir).apply(bench, savedir=savedir)
                else:
                    bench(savedir=savedir)
            else:
                bench(savedir=savedir)
            return func(*args, **kwargs)

        return wrapper

    return decorator
