from collections import OrderedDict

import numpy as np
import pytest
from matplotlib import axes, figure

from jijbench.visualization.figure.timeseries import TimeSeries

params = {
    "list case": ("data", [1, 2], [3, 4]),
    "np.ndarray case": ("data", np.array([1, 2]), np.array([3, 4])),
}


@pytest.mark.parametrize(
    "label, plot_x, plot_y",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_add_data(label, plot_x, plot_y):
    timeseries = TimeSeries()
    timeseries.add_data(label, plot_x, plot_y)

    assert timeseries.data == OrderedDict([("data", ([1, 2], [3, 4]))])


def test_timeseries_add_data_not_same_length():
    timeseries = TimeSeries()

    with pytest.raises(ValueError):
        timeseries.add_data("data", [1, 2], [3, 4, 5])


def test_timeseries_fig_ax_attribute():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


def test_timeseries_fig_ax_attribute_before_show():
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])

    with pytest.raises(AttributeError):
        timeseries.fig_ax


def test_timeseries_show_no_plot_data():
    timeseries = TimeSeries()
    with pytest.raises(RuntimeError):
        timeseries.show()


params = {
    "give argument": ("title", "title"),
    "default": (None, "time series"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_title(title, expect):
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(title=title)
    fig, ax = timeseries.fig_ax

    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


def test_timeseries_show_x_and_y():
    x1, y1 = [1, 2], [3, 4]
    x2, y2 = [5, 6], [7, 8]

    timeseries = TimeSeries()
    timeseries.add_data("data1", x1, y1)
    timeseries.add_data("data2", x2, y2)
    timeseries.show()
    fig, ax = timeseries.fig_ax

    assert (ax.get_lines()[0].get_xdata() == np.array(x1)).all()
    assert (ax.get_lines()[0].get_ydata() == np.array(y1)).all()
    assert (ax.get_lines()[1].get_xdata() == np.array(x2)).all()
    assert (ax.get_lines()[1].get_ydata() == np.array(y2)).all()


def test_timeseries_show_arg_figsize():
    figwidth, figheight = 8, 4

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(figsize=tuple([figwidth, figheight]))
    fig, ax = timeseries.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_timeseries_show_arg_color_list():
    color_list = ["r", "#e41a1c"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(color_list=color_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_color() == "r"
    assert ax.get_lines()[1].get_color() == "#e41a1c"


def test_timeseries_show_arg_color_list_invalid_length():
    color_list = ["r", "g", "b"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(color_list=color_list)


params = {
    "give argument": ([0.5, 0.7], 0.5, 0.7),
    "default": (None, 1.0, 1.0),
}


@pytest.mark.parametrize(
    "alpha_list, expect_alpha_1, expect_alpha_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_alpha_list(alpha_list, expect_alpha_1, expect_alpha_2):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(alpha_list=alpha_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_alpha() == expect_alpha_1
    assert ax.get_lines()[1].get_alpha() == expect_alpha_2


def test_timeseries_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(alpha_list=alpha_list)


params = {
    "give argument": (["-", "--"], "-", "--"),
    "default": (None, "-", "-"),
}


@pytest.mark.parametrize(
    "linestyle_list, expect_linestyle_1, expect_linestyle_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_linestyle_list(
    linestyle_list, expect_linestyle_1, expect_linestyle_2
):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(linestyle_list=linestyle_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_linestyle() == expect_linestyle_1
    assert ax.get_lines()[1].get_linestyle() == expect_linestyle_2


def test_timeseries_show_arg_linestyle_list_invalid_length():
    linestyle_list = ["--", "--", "--"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(linestyle_list=linestyle_list)


params = {
    "give argument": (["v", "d"], "v", "d"),
    "default": (None, "o", "o"),
}


@pytest.mark.parametrize(
    "marker_list, expect_marker_1, expect_marker_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_marker_list(marker_list, expect_marker_1, expect_marker_2):
    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])
    timeseries.show(marker_list=marker_list)
    fig, ax = timeseries.fig_ax

    assert ax.get_lines()[0].get_marker() == expect_marker_1
    assert ax.get_lines()[1].get_marker() == expect_marker_2


def test_timeseries_show_arg_marker_list_invalid_length():
    marker_list = ["v", "v", "v"]

    timeseries = TimeSeries()
    timeseries.add_data("data0", [1, 2], [3, 4])
    timeseries.add_data("data1", [1, 2], [3, 4])

    with pytest.raises(ValueError):
        timeseries.show(marker_list=marker_list)


params = {
    "give argument": ("xlabel", "xlabel"),
    "default": (None, "time"),
}


@pytest.mark.parametrize(
    "xlabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_timeseries_show_arg_xlabel(xlabel, expect):
    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(xlabel=xlabel)
    fig, ax = timeseries.fig_ax

    assert ax.get_xlabel() == expect


def test_timeseries_show_arg_ylabel():
    ylabel = "ylabel"

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(ylabel=ylabel)
    fig, ax = timeseries.fig_ax

    assert ax.get_ylabel() == "ylabel"


def test_timeseries_show_arg_xticks():
    xticks = [1.0, 1.5, 2.0]

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(xticks=xticks)
    fig, ax = timeseries.fig_ax

    assert (ax.get_xticks() == np.array([1.0, 1.5, 2.0])).all()


def test_timeseries_show_arg_yticks():
    yticks = [3.0, 3.5, 4.0]

    timeseries = TimeSeries()
    timeseries.add_data("data", [1, 2], [3, 4])
    timeseries.show(yticks=yticks)
    fig, ax = timeseries.fig_ax

    assert (ax.get_yticks() == np.array([3.0, 3.5, 4.0])).all()
