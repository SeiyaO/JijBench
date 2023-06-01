import os
import shutil

import jijmodeling as jm
import numpy as np
import pandas as pd
import pytest

import jijbench as jb
from jijbench.exceptions.exceptions import UserFunctionFailedError
from jijbench.visualization import MetricsParallelPlot


@pytest.fixture(scope="function", autouse=True)
def pre_post_process():
    # preprocess
    yield
    # postprocess
    norm_path = os.path.normcase("./.jb_results")
    if os.path.exists(norm_path):
        shutil.rmtree(norm_path)


def fig_contain_target_data(fig, label, values):
    for data in fig.data[0].dimensions:
        if data["label"] == label and (data["values"] == values).all():
            return True
    return False


def test_metrics_plot_parallelplot_num_feasible(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{}, {}]},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    expect_value_0 = result.table["num_feasible"].values[0]
    expect_value_1 = result.table["num_feasible"].values[1]
    expect_arr = np.array(expect_value_0, expect_value_1)

    assert fig_contain_target_data(fig, "num_feasible", expect_arr)


def test_metrics_plot_parallelplot_samplemean_objective(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{}, {}]},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    def calc_expect_samplemean_objective(result, idx):
        obj = result.table["objective"].values[idx]
        occ = result.table["num_occurrences"].values[idx]
        return np.sum(occ * obj) / np.sum(occ)

    expect_mean_0 = calc_expect_samplemean_objective(result, 0)
    expect_mean_1 = calc_expect_samplemean_objective(result, 1)
    expect_arr = np.array(expect_mean_0, expect_mean_1)

    assert fig_contain_target_data(fig, "samplemean_objective", expect_arr)


def test_metrics_plot_parallelplot_samplemean_violations(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    def calc_result_samplemean_violations(result, idx, constraint_name):
        violation = result.table[f"{constraint_name}_violations"].values[idx]
        occ = result.table["num_occurrences"].values[idx]
        return np.sum(occ * violation) / np.sum(occ)

    expect_mean_onehot1 = calc_result_samplemean_violations(result, 0, "onehot1")
    expect_mean_onehot2 = calc_result_samplemean_violations(result, 0, "onehot2")
    expect_mean_total = expect_mean_onehot1 + expect_mean_onehot2

    assert fig_contain_target_data(
        fig, "samplemean_onehot1_violations", np.array([expect_mean_onehot1])
    )

    assert fig_contain_target_data(
        fig, "samplemean_onehot2_violations", np.array([expect_mean_onehot2])
    )

    assert fig_contain_target_data(
        fig, "samplemean_total_violations", expect_mean_total
    )


def test_metrics_plot_parallelplot_samplemean_violations_no_constraint(
    jm_sampleset_no_constraint: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset_no_constraint

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    # samplemean_total_violationsが含まれていないことを確認する（本テストでは制約がない問題を解く状況を想定しているため）
    def no_samplemean_total_violations(fig):
        for data in fig.data[0].dimensions:
            if data["label"] == "samplemean_total_violations":
                return False
        return True

    assert no_samplemean_total_violations(fig)


def test_metrics_plot_parallelplot_multipliers(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    params = {
        "multipliers": [{"onehot1": 1, "onehot2": 2}, {"onehot1": 3, "onehot2": 4}]
    }

    bench = jb.Benchmark(
        params=params,
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    assert fig_contain_target_data(fig, "multipliers[onehot1]", np.array([1, 3]))
    assert fig_contain_target_data(fig, "multipliers[onehot2]", np.array([2, 4]))


def test_metrics_plot_parallelplot_arg_color_column_default(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()
    assert fig.layout.coloraxis.colorbar.title.text == "samplemean_total_violations"


def test_metrics_plot_parallelplot_arg_color_column_default_no_constraint(
    jm_sampleset_no_constraint: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset_no_constraint

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()
    assert fig.layout.coloraxis.colorbar.title.text == "samplemean_objective"


def test_metrics_plot_parallelplot_arg_color_column_default_no_obj_no_constraint(
    jm_sampleset_no_obj_no_constraint: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset_no_obj_no_constraint

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()
    assert fig.layout.coloraxis.colorbar.title.text is None


def test_metrics_plot_parallelplot_arg_color_column(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        color_column_name="samplemean_onehot1_violations"
    )

    assert fig.layout.coloraxis.colorbar.title.text == "samplemean_onehot1_violations"


def test_metrics_plot_parallelplot_arg_color_midpoint(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        color_midpoint=5,
    )

    assert fig.layout.coloraxis.cmid == 5


def test_metrics_plot_parallelplot_arg_color_midpoint_default(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={"multipliers": [{}, {}]},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    def calc_mean_total_violaions(x: pd.Series):
        num_occ = x["num_occurrences"]
        onehot1_violations = x["onehot1_violations"]
        onehot2_violations = x["onehot2_violations"]
        total_violations_mean = np.sum(
            num_occ * onehot1_violations + num_occ * onehot2_violations
        ) / np.sum(num_occ)
        return total_violations_mean

    mean_total_violations = result.table.apply(calc_mean_total_violaions, axis=1).mean()

    assert fig.layout.coloraxis.cmid == mean_total_violations


params = {
    "give_title_case": ("title", "title"),
    "default_case": (None, None),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_arg_title(title, expect, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        title=title,
    )

    assert fig.layout.title.text == expect


params = {
    "give_height_case": (1000, 1000),
    "default_case": (None, None),
}


@pytest.mark.parametrize(
    "height, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_arg_height(
    height, expect, jm_sampleset: jm.SampleSet
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        height=height,
    )

    assert fig.layout.height == expect


params = {
    "give_width_case": (1000, 1000),
    "default_case": (None, None),
}


@pytest.mark.parametrize(
    "width, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_arg_width(width, expect, jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        width=width,
    )

    assert fig.layout.width == expect


params = {
    "give_label_pos_case": ("bottom", "bottom"),
    "default_case": (None, "top"),
}


@pytest.mark.parametrize(
    "axis_label_pos, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_arg_axis_label_pos(
    axis_label_pos, expect, jm_sampleset: jm.SampleSet
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(axis_label_pos=axis_label_pos)
    assert fig.data[0].labelside == expect


params = {
    "give_fontsize_case": (20, 20),
    "default_case": (None, None),
}


def test_metrics_plot_parallelplot_arg_axis_label_pos_invalid_value(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    with pytest.raises(ValueError) as e:
        mplot.parallelplot_experiment(axis_label_pos="INVALID_VALUE")

    assert (
        str(e.value)
        == "axis_label_pos must be 'top' or 'bottom', but INVALID_VALUE is given."
    )


@pytest.mark.parametrize(
    "fontsize, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_metrics_plot_parallelplot_arg_axis_label_fontsize(
    fontsize, expect, jm_sampleset: jm.SampleSet
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(axis_label_fontsize=fontsize)
    assert fig.data[0].labelfont.size == expect


def test_metrics_plot_parallelplot_arg_additional_axes(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        additional_axes=["num_samples"],
    )

    expect_num_samples = result.table["num_samples"].values

    assert fig_contain_target_data(fig, "num_samples", expect_num_samples)


def test_metrics_plot_parallelplot_arg_additional_axes_not_number(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    with pytest.raises(TypeError):
        mplot.parallelplot_experiment(
            additional_axes=["energy"],
        )


def test_metrics_plot_parallelplot_arg_additional_axes_created_by_function(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)

    def get_max_num_occurrences(x: pd.Series) -> float:
        return np.max(x["num_occurrences"])

    def calc_samplemean_energy(x: pd.Series) -> float:
        num_occ = x["num_occurrences"]
        array = x["energy"]
        mean = np.sum(num_occ * array) / np.sum(num_occ)
        return mean

    fig = mplot.parallelplot_experiment(
        additional_axes_created_by_function={
            "max_num_occurrences": get_max_num_occurrences,
            "samplemean_energy": calc_samplemean_energy,
        },
    )

    def calc_expect_max_num_occurrences(result, idx):
        num_occ = result.table["num_occurrences"].values[idx]
        return np.max(num_occ)

    def calc_expect_samplemean_energy(result, idx):
        energy = result.table["energy"].values[idx]
        num_occ = result.table["num_occurrences"].values[idx]
        return np.sum(num_occ * energy) / np.sum(num_occ)

    expect_max_num_occurrences = calc_expect_max_num_occurrences(result, 0)
    expect_samplemean_energy = calc_expect_samplemean_energy(result, 0)

    assert fig_contain_target_data(
        fig, "max_num_occurrences", np.array([expect_max_num_occurrences])
    )
    assert fig_contain_target_data(
        fig, "samplemean_energy", np.array([expect_samplemean_energy])
    )


def test_metrics_plot_parallelplot_arg_additional_axes_created_by_function_return_not_number(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()

    def get_max_num_occurrences_return_not_number(x: pd.Series) -> str:
        return str(np.max(x["num_occurrences"]))

    mplot = MetricsParallelPlot(result)

    with pytest.raises(TypeError):
        mplot.parallelplot_experiment(
            additional_axes_created_by_function={
                "max_num_occurrences": get_max_num_occurrences_return_not_number
            }
        )


def test_metrics_plot_parallelplot_arg_additional_axes_created_by_function_error_in_func(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()

    def error_func(x: pd.Series) -> str:
        raise RuntimeError

    mplot = MetricsParallelPlot(result)
    with pytest.raises(UserFunctionFailedError):
        mplot.parallelplot_experiment(
            additional_axes_created_by_function={"column_name": error_func}
        )


def test_metrics_plot_parallelplot_arg_display_axes_list(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        display_axes_list=[
            "samplemean_onehot2_violations",
            "samplemean_onehot1_violations",
        ]
    )

    assert len(fig.data[0].dimensions) == 2
    assert fig.data[0].dimensions[0].label == "samplemean_onehot2_violations"
    assert fig.data[0].dimensions[1].label == "samplemean_onehot1_violations"


def test_metrics_plot_parallelplot_property_parallelplot_axes_list(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment()

    expect_name_list = [dimension.label for dimension in fig.data[0].dimensions]

    assert mplot.parallelplot_axes_list == expect_name_list


def test_metrics_plot_parallelplot_property_parallelplot_axes_list_before_call_parallelplot(
    jm_sampleset: jm.SampleSet,
):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)

    assert mplot.parallelplot_axes_list == []


def test_metrics_plot_parallelplot_arg_rename_map(jm_sampleset: jm.SampleSet):
    def func() -> jm.SampleSet:
        return jm_sampleset

    bench = jb.Benchmark(
        params={},
        solver=[func],
    )
    result = bench()
    mplot = MetricsParallelPlot(result)
    fig = mplot.parallelplot_experiment(
        rename_map={
            "samplemean_onehot1_violations": "Samplemean Onehot1 Violations",
            "samplemean_onehot2_violations": "Samplemean Onehot2 Violations",
            "samplemean_total_violations": "Samplemean Total Violations",
            "hoge": "Hoge",
        }
    )

    def calc_result_samplemean_violations(result, idx, constraint_name):
        violation = result.table[f"{constraint_name}_violations"].values[idx]
        occ = result.table["num_occurrences"].values[idx]
        return np.sum(occ * violation) / np.sum(occ)

    expect_mean_onehot1 = calc_result_samplemean_violations(result, 0, "onehot1")
    expect_mean_onehot2 = calc_result_samplemean_violations(result, 0, "onehot2")
    expect_mean_total = expect_mean_onehot1 + expect_mean_onehot2

    assert fig_contain_target_data(
        fig, "Samplemean Onehot1 Violations", np.array([expect_mean_onehot1])
    )

    assert fig_contain_target_data(
        fig, "Samplemean Onehot2 Violations", np.array([expect_mean_onehot2])
    )

    assert fig_contain_target_data(
        fig, "Samplemean Total Violations", expect_mean_total
    )
