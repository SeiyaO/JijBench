from collections import OrderedDict

import numpy as np
import pytest
from matplotlib import axes, figure, patches

from jijbench.visualization import Schedule

params = {
    "list case": ("data", [1, 2], [3, 4], [5.5, 6.6]),
    "np.ndarray case": (
        "data",
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5.5, 6.6]),
    ),
}


@pytest.mark.parametrize(
    "task_label, workers, start_times, time_lengths",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_add_data(task_label, workers, start_times, time_lengths):
    schedule = Schedule()
    schedule.add_data(task_label, workers, start_times, time_lengths)

    assert schedule.data == OrderedDict([("data", ([1, 2], [3, 4], [5.5, 6.6]))])


def test_schedule_add_data_attribute_workers():
    schedule = Schedule()
    schedule.add_data("data1", [1, 2], [3, 4], [2, 4])
    schedule.add_data("data2", [2, 3], [7, 8], [9, 10])

    assert schedule.workers == [1, 2, 3]


params = {
    "workers and start_times are different": ("data", [1, 2], [3, 4, 5], [5.5, 6.6]),
    "workers and time_lengths are different": (
        "data",
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5.5, 6.6, 7.7]),
    ),
}


@pytest.mark.parametrize(
    "task_label, workers, start_times, time_lengths",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_add_data_not_same_length(
    task_label, workers, start_times, time_lengths
):
    schedule = Schedule()

    with pytest.raises(ValueError):
        schedule.add_data(task_label, workers, start_times, time_lengths)


def test_schedule_fig_ax_attribute():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)
    schedule.show()
    fig, ax = schedule.fig_ax

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


def test_schedule_fig_ax_attribute_before_show():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)

    with pytest.raises(AttributeError):
        schedule.fig_ax


def test_schedule_show_no_plot_data():
    schedule = Schedule()
    with pytest.raises(RuntimeError):
        schedule.show()


params = {
    "give argument": ("title", "title"),
    "default": (None, "schedule"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_title(title, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [3, 4], [5, 6])
    schedule.show(title=title)
    fig, ax = schedule.fig_ax

    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


def test_schedule_show_bar():
    workers1, start_times1, time_lengths1 = [1, 2], [1, 2], [3, 4]
    workers2, start_times2, time_lengths2 = [2], [1], [1]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)

    schedule.show()
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected bar information
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_show_bar_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_1 = obj.get_center()
            if (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                success_show_bar_1 = True
    assert success_show_bar_1

    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times1[1] + (time_length1[1] / 2), workers1[1]])
    success_show_bar_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_2 = obj.get_center()
            if (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                success_show_bar_2 = True
    assert success_show_bar_2

    expect_center_3 = np.array(
        [1.5, 2]
    )  # np.array([start_times2[0] + (time_length2[0] / 2), workers2[0]])
    success_show_bar_3 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_center_3 = obj.get_center()
            if (np.abs(actual_center_3 - expect_center_3) < 0.0001).all():
                success_show_bar_3 = True
    assert success_show_bar_3


def test_schedule_show_text():
    workers, start_times, time_lengths = [1, 2], [1, 2], [3, 4]

    schedule = Schedule()
    schedule.add_data("data", workers, start_times, time_lengths)
    schedule.show()
    fig, ax = schedule.fig_ax

    # Check that the ax.texts contain the expected text information
    expect_text_1 = "3"  # time_length[0]
    expect_center_1 = (2.5, 1)  # (start_times[0] + (time_length[0] / 2), workers[0])
    success_text_1 = False
    for obj in ax.texts:
        actual_text_1 = obj.get_text()
        actual_center_1 = obj.get_position()
        if not (actual_text_1 == expect_text_1):
            continue
        if not (actual_center_1 == expect_center_1):
            continue
        success_text_1 = True
    assert success_text_1

    expect_text_2 = "4"  # time_length[1]
    expect_center_2 = (4, 2)  # (start_times[1] + (time_length[1] / 2), workers[1])
    success_text_2 = False
    for obj in ax.texts:
        actual_text_2 = obj.get_text()
        actual_center_2 = obj.get_position()
        if not (actual_text_2 == expect_text_2):
            continue
        if not (actual_center_2 == expect_center_2):
            continue
        success_text_2 = True
    assert success_text_2


def test_schedule_show_arg_figsize():
    figwidth, figheight = 8, 4

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(figsize=tuple([figwidth, figheight]))
    fig, ax = schedule.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_schedule_show_arg_color_list():
    color_list = ["red", "blue"]
    workers1, start_times1, time_lengths1 = [1], [1], [3]
    workers2, start_times2, time_lengths2 = [2], [2], [4]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)
    schedule.show(color_list=color_list)
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected color information
    expect_color_1 = np.array([1.0, 0.0, 0.0])  # red
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_coloring_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_color_1 = obj.get_facecolor()[:-1]
            actual_center_1 = obj.get_center()
            if not (actual_color_1 == expect_color_1).all():
                continue
            if not (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                continue
            success_coloring_1 = True
    assert success_coloring_1

    expect_color_2 = np.array([0.0, 0.0, 1.0])  # blue
    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times2[0] + (time_length2[0] / 2), workers2[0]])
    success_coloring_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_color_2 = obj.get_facecolor()[:-1]
            actual_center_2 = obj.get_center()
            if not (actual_color_2 == expect_color_2).all():
                continue
            if not (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                continue
            success_coloring_2 = True
    assert success_coloring_2


def test_schedule_show_arg_color_list_invalid_length():
    color_list = ["r", "g", "b"]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(color_list=color_list)


params = {
    "give argument": ([0.3, 0.7], 0.3, 0.7),
    "default": (None, 0.5, 0.5),
}


@pytest.mark.parametrize(
    "alpha_list, expect_alpha_1, expect_alpha_2",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_alpha_list(alpha_list, expect_alpha_1, expect_alpha_2):
    workers1, start_times1, time_lengths1 = [1], [1], [3]
    workers2, start_times2, time_lengths2 = [2], [2], [4]

    schedule = Schedule()
    schedule.add_data("data1", workers1, start_times1, time_lengths1)
    schedule.add_data("data2", workers2, start_times2, time_lengths2)
    schedule.show(alpha_list=alpha_list)
    fig, ax = schedule.fig_ax

    # Check that the children of ax contain the expected alpha information
    expect_center_1 = np.array(
        [2.5, 1]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_alpha_1 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_alpha_1 = obj.get_alpha()
            actual_center_1 = obj.get_center()
            if not actual_alpha_1 == expect_alpha_1:
                continue
            if not (np.abs(actual_center_1 - expect_center_1) < 0.0001).all():
                continue
            success_alpha_1 = True
    assert success_alpha_1

    expect_center_2 = np.array(
        [4, 2]
    )  # np.array([start_times1[0] + (time_length1[0] / 2), workers1[0]])
    success_alpha_2 = False
    for obj in ax.get_children():
        if type(obj) == patches.Rectangle:
            actual_alpha_2 = obj.get_alpha()
            actual_center_2 = obj.get_center()
            if not actual_alpha_2 == expect_alpha_2:
                continue
            if not (np.abs(actual_center_2 - expect_center_2) < 0.0001).all():
                continue
            success_alpha_2 = True
    assert success_alpha_2


def test_schedule_show_arg_alpha_list_invalid_length():
    alpha_list = [0.1, 0.1, 0.1]

    schedule = Schedule()
    schedule.add_data("data0", [1, 2], [1, 2], [3, 4])
    schedule.add_data("data1", [1, 2], [1, 2], [3, 4])

    with pytest.raises(ValueError):
        schedule.show(alpha_list=alpha_list)


params = {
    "give argument": ("xlabel", "xlabel"),
    "default": (None, "time"),
}


@pytest.mark.parametrize(
    "xlabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_xlabel(xlabel, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xlabel=xlabel)
    fig, ax = schedule.fig_ax

    assert ax.get_xlabel() == expect


params = {
    "give argument": ("ylabel", "ylabel"),
    "default": (None, "worker"),
}


@pytest.mark.parametrize(
    "ylabel, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_ylabel(ylabel, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(ylabel=ylabel)
    fig, ax = schedule.fig_ax

    assert ax.get_ylabel() == expect


def test_schedule_show_arg_xticks():
    xticks = [1, 2, 3, 4, 5, 6]

    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(xticks=xticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_xticks() == np.array([1, 2, 3, 4, 5, 6])).all()


params = {
    "give argument": ([1, 2, 3], np.array([1, 2, 3])),
    "default": (None, np.array([1, 2])),
}


@pytest.mark.parametrize(
    "yticks, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_schedule_show_arg_yticks(yticks, expect):
    schedule = Schedule()
    schedule.add_data("data", [1, 2], [1, 2], [3, 4])
    schedule.show(yticks=yticks)
    fig, ax = schedule.fig_ax

    assert (ax.get_yticks() == expect).all()
