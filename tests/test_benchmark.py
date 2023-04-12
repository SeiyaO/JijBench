import os
import shutil
import typing as tp
from unittest.mock import MagicMock

import jijmodeling as jm
import jijzept as jz
import pandas as pd
import pytest

import jijbench as jb


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def f1():
    return "Hello, World!"


def f2(i: int) -> int:
    return i**2


def f3(i: int) -> tuple[int, int]:
    return i + 1, i + 2


def f4(i: int, j: int = 1) -> int:
    return i + j


@pytest.mark.parametrize(
    "solver, params",
    [
        (f1, {}),
        (f2, {"i": [1, 2]}),
        (f3, {"i": [1, 2]}),
        (f4, {"i": [1, 2], "j": [1, 2]}),
    ],
)
def test_benchmark(
    solver: tp.Callable[..., tp.Any], params: dict[str, tp.Iterable[tp.Any]]
):
    bench = jb.Benchmark(solver=solver, params=params, name="test")

    res = bench(autosave=True)
    assert isinstance(res, jb.Experiment)

    actual = res.response_table

    solver_args = [{pi.name: pi.data for pi in p} for p in bench.params]
    expected = pd.DataFrame(
        map(lambda x: solver(**x), solver_args),
        index=actual.index,
        columns=actual.columns,
    )

    assert actual.equals(expected)

    op = res.operator
    assert op is not None
    assert isinstance(op.inputs[0], jb.Experiment)
    assert isinstance(op.inputs[1], jb.Experiment)


def test_benchmark_for_jijzept_sampler(
    sample_model: MagicMock,
    sa_sampler: jz.JijSASampler,
    knapsack_problem: jm.Problem,
    knapsack_instance_data: jm.PH_VALUES_INTERFACE,
):
    bench = jb.construct_benchmark_for(
        sa_sampler,
        [(knapsack_problem, knapsack_instance_data)],
        {"num_reads": [1, 2]},
    )
    res = bench(autosave=False)

    assert sample_model.call_count == 2
    assert len(sample_model.call_args_list) == 2
    sample_model.assert_called_with(
        sa_sampler,
        model=knapsack_problem,
        feed_dict=knapsack_instance_data,
        num_reads=2,
    )

    table = res.table.reset_index()
    assert table.loc[0, "num_samples"] == 10
    assert table.loc[0, "num_feasible"] == 7


def test_benchmark_for_jijzept_sampler_with_multi_models(
    sample_model: MagicMock,
    sa_sampler: jz.JijSASampler,
    knapsack_problem: jm.Problem,
    knapsack_instance_data: jm.PH_VALUES_INTERFACE,
    tsp_problem: jm.Problem,
    tsp_instance_data: jm.PH_VALUES_INTERFACE,
):
    models = [
        (knapsack_problem, knapsack_instance_data),
        (tsp_problem, tsp_instance_data),
    ]
    bench = jb.construct_benchmark_for(
        sa_sampler,
        models,
        {
            "search": [True, False],
            "num_search": [5],
        },
    )
    res = bench(autosave=False)

    assert sample_model.call_count == 4
    assert len(sample_model.call_args_list) == 4

    sample_model.assert_any_call(
        sa_sampler,
        model=knapsack_problem,
        feed_dict=knapsack_instance_data,
        search=True,
        num_search=5,
    )
    sample_model.assert_any_call(
        sa_sampler,
        model=tsp_problem,
        feed_dict=tsp_instance_data,
        search=False,
        num_search=5,
    )

    table = res.table.reset_index()
    assert table.loc[0, "num_samples"] == 10
    assert table.loc[0, "num_feasible"] == 7


def test_benchmark_for_jijzept_sampler_using_params(
    onehot_problem: jm.Problem, jm_sampleset: jm.SampleSet
):
    def f(problem, instance_data, **kwargs) -> jm.SampleSet:
        if not isinstance(problem, jm.Problem):
            raise TypeError
        if not isinstance(instance_data, dict):
            raise TypeError
        return jm_sampleset

    instance_data = {"d": [1 for _ in range(10)]}
    instance_data["d"][0] = -1

    bench = jb.Benchmark(
        solver=f,
        params={
            "num_reads": [1, 2],
            "num_sweeps": [10],
            "problem": [onehot_problem],
            "instance_data": [instance_data],
        },
    )
    res = bench(autosave=False)

    # assert res.table["problem_name"][0] == "problem"
    # assert res.table["instance_data_name"][0] == "Unnamed[0]"


def test_apply_benchmark():
    def func(x):
        return x

    bench = jb.Benchmark(
        solver=func,
        params={"x": [1, 2]},
    )

    experiment = jb.Experiment(name=jb.ID().data)
    res = experiment.apply(bench)
    columns = res.table.columns

    assert isinstance(res, jb.Experiment)
    assert "solver_return[0]" in columns

    op1 = res.operator

    assert op1 is not None
    assert isinstance(op1, jb.Benchmark)
    assert isinstance(op1.inputs[0], jb.Experiment)
    assert len(op1.inputs) == 1
    assert op1.inputs[0].table.empty


def test_benchmark_params_table():
    def func(x):
        return x

    bench = jb.Benchmark(
        solver=func,
        params={"x": [1, 2]},
    )

    res = bench()


def test_benchmark_with_multi_return_solver():
    def func():
        return "a", 1

    bench = jb.Benchmark(solver=func, params={"num_reads": [1, 2], "num_sweeps": [10]})
    res = bench()

    assert len(res.table) == 2
    assert res.table["solver_return[0]"][0] == "a"
    assert res.table["solver_return[1]"][0] == 1.0


# def test_benchmark_with_custom_solver_by_sync_False():
#     def func():
#         return "a", 1
#
#     bench = jb.Benchmark({"num_reads": [1, 2], "num_sweeps": [10]}, solver=func)
#     with pytest.raises(ConcurrentFailedError):
#         bench.run(sync=False)


def test_benchmark_with_callable_args():
    def f(x):
        return x**2

    def rap_solver(x, f):
        return f(x)

    bench = jb.Benchmark(
        solver=rap_solver,
        params={
            "x": [1, 2, 3],
            "f": [f],
        },
    )

    res = bench()

    # assert sample_model.__name__ in columns
    # assert isinstance(res.table[sample_model.__name__][0], str)


@pytest.mark.parametrize(
    "x, y, z, kwargs, expected",
    [
        (1, None, None, {}, "1"),
        (1, 1, None, {}, "2"),
        (1, None, 1, {}, "2"),
        (1, 1, 1, {}, "3"),
        (1, 1, None, {"extra": "!"}, "2!"),
    ],
)
def test_benchmark_by_checkpoint(x, y, z, kwargs, expected):
    benchmark_id = "example_checkpoint"

    @jb.checkpoint(name=benchmark_id, savedir="./checkpoint")
    def pos_or_kw(x: int, y: int = 0, z: int = 0, **kwargs) -> str:
        extra = "".join(kwargs.values())
        return f"{x + y + z}{extra}"

    @jb.checkpoint(name=benchmark_id, savedir="./checkpoint")
    def kw_only(x: int, *, y: int = 0, z: int = 0, **kwargs) -> str:
        extra = "".join(kwargs.values())
        return f"{x + y + z}{extra}"

    if y and z:
        ret1 = pos_or_kw(x, y, z, **kwargs)
        ret2 = kw_only(x, y=y, z=z, **kwargs)
    elif y and not z:
        ret1 = pos_or_kw(x, y, **kwargs)
        ret2 = kw_only(x, y=y, **kwargs)
    elif not y and z:
        ret1 = pos_or_kw(x, z=z, **kwargs)
        ret2 = kw_only(x, z=z, **kwargs)
    else:
        ret1 = pos_or_kw(x, **kwargs)
        ret2 = kw_only(x, **kwargs)

    assert ret1 == expected
    assert ret1 == ret2

    bench = jb.load(benchmark_id, savedir="./checkpoint")
    assert (bench.table["solver_return[0]"][-2:] == expected).all()

    shutil.rmtree("./checkpoint")
