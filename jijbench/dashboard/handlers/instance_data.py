from __future__ import annotations

import glob
import pathlib
import typing as tp

import streamlit as st

from jijbench.consts.default import DEFAULT_PROBLEM_NAMES, DEFAULT_RESULT_DIR


class InstanceDataDir:
    num_files_to_display = 5

    def __init__(self) -> None:
        self._node_map: dict[str, tp.Any] = {}
        for problem_name in self._problem_names:
            node: dict[str, tp.Any] = {
                "label": problem_name,
                "value": problem_name,
                "children": [],
            }
            for size in ["small", "medium", "large"]:
                files = glob.glob(
                    f"{self.base_dir}/{size}/{problem_name}/**/*.json", recursive=True
                )
                files.sort()
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


class InstanceDataHandler:
    def __init__(self, base_data_dir: pathlib.Path = DEFAULT_RESULT_DIR) -> None:
        self.base_data_dir = base_data_dir

    def on_add(self, session: Session) -> None:
        problem_name = session.state.input_problem_name
        instance_data_name = session.state.uploaded_instance_data_name
        instance_data_dir = session.state.instance_data_dir

        file = f"{self.base_data_dir}/{problem_name}/{instance_data_name}"
        if problem_name in instance_data_dir.node_map:
            if file not in [
                child["value"]
                for child in instance_data_dir.node_map[problem_name]["children"]
            ]:
                instance_data_dir.node_map[problem_name]["children"] += [
                    {"label": instance_data_name, "value": file}
                ]
        else:
            instance_data_dir.node_map[problem_name] = {
                "label": problem_name,
                "value": problem_name,
                "children": [{"label": instance_data_name, "value": file}],
            }

    def on_load(self, session: Session) -> None:
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
        fig = _plot_histogram_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_box(self, data: dict[str, list[int | float]]) -> None:
        fig = _plot_box_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_violin(self, data: dict[str, list[int | float]]) -> None:
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
        with open(file, "r") as f:
            ins_d = rapidjson.load(f)
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
