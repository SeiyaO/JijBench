from __future__ import annotations

import inspect
import pathlib
import typing as tp
from functools import wraps

import jijbench as jb
from jijbench.consts.path import DEFAULT_RESULT_DIR


def checkpoint(
    name: str | None = None, savedir: str | pathlib.Path = DEFAULT_RESULT_DIR
):
    def decorator(func: tp.Callable[..., tp.Any]):
        @wraps(func)
        def wrapper(*args: tp.Any, **kwargs: tp.Any):
            params = {}
            signature = inspect.signature(func)
            pos_arg_index = 0
            kw_arg_index = 0
            for k, v in signature.parameters.items():
                if v.kind == 1:
                    if k in kwargs:
                        params[k] = [kwargs[k]]
                        kw_arg_index += 1
                    else:
                        params[k] = [args[pos_arg_index]]
                        pos_arg_index += 1
                elif v.kind == 2:
                    if pos_arg_index < len(args):
                        for arg in args[pos_arg_index:]:
                            params[f"args[{pos_arg_index}]"] = [arg]
                            pos_arg_index += 1
                elif v.kind == 3:
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
                    jb.load(name).apply(bench, savedir=savedir)
                else:
                    bench(savedir=savedir)
            else:
                bench(savedir=savedir)

        return wrapper

    return decorator
