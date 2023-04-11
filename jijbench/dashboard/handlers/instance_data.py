from __future__ import annotations

import pathlib
import typing as tp

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from jijbench.consts.default import DEFAULT_PROBLEM_NAMES
from jijbench.datasets.model import InstanceDataFileStorage

if tp.TYPE_CHECKING:
    from jijbench.dashboard.session import Session


class InstanceDataDirTree:
    """
    A class representing a directory tree of instance data files.

    Attributes:
        num_files_to_display (int): The number of files to display.
        node_map (dict[str, tp.Any]): A property to get or set the node map.
        nodes (list[dict[str, tp.Any]]): A property to get the list of nodes.
        problem_names (list[str]): A property to get or set the list of problem names.
    """

    def __init__(self) -> None:
        self._node_map: dict[str, tp.Any] = {}
        self._num_files_to_display = 5
        self._problem_names = DEFAULT_PROBLEM_NAMES

        for problem_name in DEFAULT_PROBLEM_NAMES:
            storage = InstanceDataFileStorage(problem_name)

            node: dict[str, tp.Any] = {
                "label": problem_name,
                "value": problem_name,
                "children": [],
            }
            for size in ["small", "medium", "large"]:
                files = storage.get_files(size, self.num_files_to_display)
                node["children"] += [
                    {
                        "label": size,
                        "value": f"{problem_name}&{size}",
                        "children": [
                            {
                                "label": f"sample-{i + 1:03}",
                                "value": file,
                            }
                            for i, file in enumerate(files[: self.num_files_to_display])
                        ],
                    },
                ]
            self._node_map[problem_name] = node

    @property
    def node_map(self) -> dict[str, tp.Any]:
        return self._node_map

    @node_map.setter
    def node_map(self, node_map: dict[str, tp.Any]) -> None:
        self._node_map = node_map

    @property
    def nodes(self) -> list[dict[str, tp.Any]]:
        return list(self._node_map.values())

    @property
    def problem_names(self) -> list[str]:
        return self._problem_names

    @problem_names.setter
    def problem_names(self, problem_names: list[str]) -> None:
        self._problem_names = problem_names

    @property
    def num_files_to_display(self) -> int:
        return self._num_files_to_display

    @num_files_to_display.setter
    def num_files_to_display(self, num_files_to_display: int) -> None:
        self._num_files_to_display = num_files_to_display


class InstanceDataHandler:
    """
    A class to handle instance data files, manage their loading, and visualize them using various plot types.

    """

    def on_add(self, session: Session) -> None:
        """Adds a new instance data file to the directory tree.

        Args:
            session (Session): The current session.
        """
        problem_name = session.state.input_problem_name
        instance_data_name = session.state.uploaded_instance_data_name
        logdir = session.state.logdir
        instance_data_dir_tree = session.state.instance_data_dir_tree

        file = f"{logdir}/{problem_name}/{instance_data_name}"
        if problem_name in instance_data_dir_tree.node_map:
            if file not in [
                child["value"]
                for child in instance_data_dir_tree.node_map[problem_name]["children"]
            ]:
                instance_data_dir_tree.node_map[problem_name]["children"] += [
                    {"label": instance_data_name, "value": file}
                ]
        else:
            instance_data_dir_tree.node_map[problem_name] = {
                "label": problem_name,
                "value": problem_name,
                "children": [{"label": instance_data_name, "value": file}],
            }

        instance_data_dir_tree.problem_names += [problem_name]

    def on_load(self, session: Session) -> None:
        """Loads selected instance data files and displays the chosen plot type.

        Args:
            session (Session): The current session.
        """
        files = session.state.selected_instance_data_files
        data_map = _load_instance_data(files)

        name = session.state.selected_instance_data_name
        fig_type = session.state.selected_figure_for_instance_data
        if name:
            key = name.split("/")[-1]
            if fig_type == "Histogram":
                self.on_select_histogram(data_map[key])
            elif fig_type == "Box":
                self.on_select_box(data_map[key])
            elif fig_type == "Violin":
                self.on_select_violin(data_map[key])

    def on_select_histogram(self, data: dict[str, list[int | float]]) -> None:
        """Creates a histogram plot of the given data.

        Args:
            data (dict[str, list[int | float]]): The instance data to be visualized.
        """
        fig = _plot_histogram_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_box(self, data: dict[str, list[int | float]]) -> None:
        """Creates a box plot of the given data.

        Args:
            data (dict[str, list[int | float]]): The instance data to be visualized.
        """
        fig = _plot_box_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_violin(self, data: dict[str, list[int | float]]) -> None:
        """Display a violin plot for the given instance data.


        Args:
            data (dict[str, list[int | float]]): The instance data to be visualized.
        """
        fig = _plot_violin_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def _load_instance_data(files: list[str]) -> dict[str, dict[str, list[int | float]]]:
    def _flatten(data: int | float | list[tp.Any]) -> list[tp.Any]:
        if isinstance(data, list):
            return [xij for xi in data for xij in _flatten(xi)]
        else:
            return [data]

    data = {}
    for file in files:
        ins_d = InstanceDataFileStorage.load(file)
        data[pathlib.Path(file).name] = {k: _flatten(v) for k, v in ins_d.items()}
    return data


def _plot_histogram_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
    fig = make_subplots(rows=len(data), cols=1)
    for i, (k, v) in enumerate(data.items()):
        v = np.array(v)
        if len(np.unique(v)) == 1:
            xmin, xmax = v[0] - 10, v[0] + 10
        else:
            xmin, xmax = v.min(), v.max()
        fig.add_trace(go.Histogram(x=v, name=k, nbinsx=100), row=i + 1, col=1)
        fig.update_xaxes(range=[xmin, xmax], row=i + 1, col=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=200 * len(data),
        width=700,
    )
    return fig


def _plot_box_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(data))
    for i, (k, v) in enumerate(data.items()):
        fig.add_trace(go.Box(y=v, name=k), row=1, col=i + 1)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _plot_violin_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(data))
    for i, (k, v) in enumerate(data.items()):
        fig.add_trace(
            go.Violin(y=v, x0=k, name=k, box_visible=True, meanline_visible=True),
            row=1,
            col=i + 1,
        )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
