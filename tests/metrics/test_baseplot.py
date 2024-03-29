import jijmodeling as jm
import numpy as np
import pandas as pd
import pytest
from matplotlib import axes, figure

import jijbench as jb
from jijbench.visualization import BasePlot, construct_experiment_from_samplesets


def simple_func_for_boxplot(x: pd.Series) -> dict:
    return {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}


def test_base_plot_boxplot_return_value(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [func]

    expect_num_fig = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    bplot = BasePlot(result)
    fig_ax_tuple = bplot.boxplot(f=simple_func_for_boxplot)
    assert len(fig_ax_tuple) == expect_num_fig
    assert type(fig_ax_tuple[0][0]) == figure.Figure
    assert type(fig_ax_tuple[0][1]) == axes.Subplot


def test_base_plot_boxplot_call_maplotlib_boxplot(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [func]

    expect_num_call = len(params["multipliers"]) * len(solver_list)

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(f=simple_func_for_boxplot)

    assert m.call_count == expect_num_call


def test_base_plot_boxplot_arg_figsize(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    figwidth, figheight = 8, 4

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    fig_ax_tuple = bplot.boxplot(
        f=simple_func_for_boxplot,
        figsize=(figwidth, figheight),
    )
    fig, ax = fig_ax_tuple[0]

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_base_plot_boxplot_arg_title(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [func]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(
        f=simple_func_for_boxplot,
        title=title,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=None),
            mocker.call("title2", fontsize=None),
        ]
    )


def test_base_plot_boxplot_arg_title_fontsize(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.figure.Figure.suptitle")

    title = ["title1", "title2"]
    title_fontsize = 12

    num_multipliers = 2
    params = {"multipliers": [{} for _ in range(num_multipliers)]}
    solver_list = [func]

    bench = jb.Benchmark(
        params=params,
        solver=solver_list,
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(
        f=simple_func_for_boxplot,
        title=title,
        title_fontsize=title_fontsize,
    )
    m.assert_has_calls(
        [
            mocker.call("title1", fontsize=title_fontsize),
            mocker.call("title2", fontsize=title_fontsize),
        ]
    )


def test_base_plot_boxplot_arg_xticklabels(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    fig_ax_tuple = bplot.boxplot(
        f=simple_func_for_boxplot,
    )
    ax = fig_ax_tuple[0][1]
    xticklabels = ax.get_xticklabels()

    assert xticklabels[0].get_text() == "data1"
    assert xticklabels[1].get_text() == "data2"


def test_base_plot_boxplot_arg_xticklabels_size(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    xticklabels_size = 10
    data = {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(
        f=simple_func_for_boxplot,
        xticklabels_size=xticklabels_size,
    )

    m.assert_called_once_with(data.keys(), size=10, rotation=None)


def test_base_plot_boxplot_arg_xticklabels_rotation(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.axes.Subplot.set_xticklabels")
    xticklabels_rotation = 10
    data = {"data1": np.array([1, 2, 3, 4]), "data2": np.array([1, 2, 3, 4])}

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(
        f=simple_func_for_boxplot,
        xticklabels_rotation=xticklabels_rotation,
    )

    m.assert_called_once_with(data.keys(), size=None, rotation=10)


def test_base_plot_boxplot_arg_ylabel(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    ylabel = "ylabel"

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    fig_ax_tuple = bplot.boxplot(
        f=simple_func_for_boxplot,
        ylabel=ylabel,
    )

    ax = fig_ax_tuple[0][1]
    assert ax.get_ylabel() == "ylabel"


def test_base_plot_boxplot_arg_ylabel_size(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.axes.Subplot.set_ylabel")

    ylabel = "ylabel"
    ylabel_size = 10

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(f=simple_func_for_boxplot, ylabel=ylabel, ylabel_size=ylabel_size)

    m.assert_called_once_with("ylabel", size=ylabel_size)


def test_base_plot_boxplot_arg_yticks_default(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.ticker.MaxNLocator.__init__", return_value=None)

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(f=simple_func_for_boxplot)

    m.assert_called_with(integer=True)


def test_base_plot_boxplot_arg_yticks(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    yticks = [0, 10]

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    fig_ax_tuple = bplot.boxplot(
        f=simple_func_for_boxplot,
        yticks=yticks,
    )
    ax = fig_ax_tuple[0][1]
    assert (ax.get_yticks() == [0, 10]).all()


def test_base_plot_boxplot_arg_kwargs(mocker, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    m = mocker.patch("matplotlib.axes.Subplot.boxplot")

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    bplot = BasePlot(result)
    bplot.boxplot(
        f=simple_func_for_boxplot,
        showmeans=True,
        meanline=True,
    )

    args, kwargs = m.call_args
    assert kwargs["showmeans"] is True
    assert kwargs["meanline"] is True
