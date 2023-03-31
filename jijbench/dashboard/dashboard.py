from __future__ import annotations

import altair as alt
import codecs
import datetime
import glob
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
import typing as tp

import jijbench as jb
import numpy as np
import pandas as pd
import pathlib
import rapidjson
import streamlit as st
import streamlit_nested_layout

from awesome_table import AwesomeTable
from awesome_table.column import Column
from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench import datasets
from jijbench.visualization.metrics.plot import MetricsPlot
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_ace import st_ace
from streamlit_elements import editor, elements, mui, lazy, sync
from streamlit_tree_select import tree_select
from typing import MutableMapping
from typing_extensions import TypeGuard


st.set_page_config(layout="wide")


class InstanceDataDir:
    jijbench_default_problem_names: list[str] = [
        "BinPacking",
        "Knapsack",
        "TSP",
        "TSPTW",
        "NurseScheduling",
    ]
    base_dir = (
        "/home/d/.venv/sandbox/lib/python3.9/site-packages/jijbench/datasets/Instances"
    )
    num_files_to_display = 5

    def __init__(self) -> None:
        self._problem_names = [
            "bin-packing",
            "knapsack",
            "nurse-scheduling",
            "travelling-salesman",
            "travelling-salesman-with-time-windows",
        ]
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
        data_map = load_instance_data(files)

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
        fig = plot_histogram_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_box(self, data: dict[str, list[int | float]]) -> None:
        fig = plot_box_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    def on_select_violin(self, data: dict[str, list[int | float]]) -> None:
        fig = plot_violin_for_instance_data(data)
        with st.container():
            st.plotly_chart(fig, use_container_width=True)


class PagingHandler:
    def on_select_page(self, session: Session) -> None:
        page = session.state.selected_page
        if page == "Instance data":
            self.on_select_instance_data(session)
        elif page == "Problem":
            self.on_select_problem(session)
        elif page == "Solver":
            self.on_select_solver(session)
        elif page == "Analysis":
            self.on_select_result(session)

    def on_select_instance_data(self, session: Session) -> None:
        session.state.selected_figure_for_instance_data = st.radio(
            "Fugure",
            options=["Histogram", "Box", "Violin"],
            horizontal=True,
        )
        options = sum(
            [
                [
                    f"{problem_name}/{pathlib.Path(f).name}"
                    for f in session.state.selected_instance_data_files
                    if problem_name in f
                ]
                for problem_name in session.state.selected_problem_names
            ],
            [],
        )
        session.state.selected_instance_data_name = st.radio(
            "Loaded instance data",
            options=options,
            horizontal=True,
        )

        if session.state.is_instance_data_loaded:
            session.plot_instance_data()

        cols = st.columns(2)
        with cols[0]:
            with st.expander("List", expanded=True):
                session.state.selected_instance_data_map = tree_select(
                    session.state.instance_data_dir.nodes,
                    check_model="leaf",
                    only_leaf_checkboxes=True,
                )
                if st.button("Load", key="load_instance_data"):
                    session.state.is_instance_data_loaded = True
                    st.experimental_rerun()

        with cols[1]:
            with st.expander("Upload", expanded=True):
                with st.form("new_instance_data"):
                    session.state.input_problem_name = st.text_input(
                        "Input problem name"
                    )
                    byte_stream = st.file_uploader(
                        "Upload your instance data", type=["json"]
                    )
                    if byte_stream:
                        session.state.uploaded_instance_data_name = byte_stream.name
                        ins_d = rapidjson.loads(byte_stream.getvalue())
                    if st.form_submit_button("Submit"):
                        if byte_stream:
                            session.add_instance_data()
                            st.experimental_rerun()

    def on_select_solver(self, session: Session) -> None:
        # with elements("monaco_editors"):

        #     # Streamlit Elements embeds Monaco code and diff editor that powers Visual Studio Code.
        #     # You can configure editor's behavior and features with the 'options' parameter.
        #     #
        #     # Streamlit Elements uses an unofficial React implementation (GitHub links below for
        #     # documentation).

        #     from streamlit_elements import editor

        #     if "content" not in st.session_state:
        #         st.session_state.content = "Default value"

        #     mui.Typography("Content: ", st.session_state.content)

        #     def update_content(value):
        #         st.session_state.content = value

        #     editor.Monaco(
        #         height=300,
        #         defaultValue=st.session_state.content,
        #         onChange=lazy(update_content),
        #     )

        #     mui.Button("Update content", onClick=sync())

        #     editor.MonacoDiff(
        #         original="Happy Streamlit-ing!",
        #         modified="Happy Streamlit-in' with Elements!",
        #         height=300,
        #     )
        pass

    def on_select_problem(self, session: Session) -> None:
        def is_callable(obj: tp.Any, name: str) -> TypeGuard[tp.Callable[..., tp.Any]]:
            check_type(name, obj, tp.Callable[..., tp.Any])
            return True

        def get_function_from_code(code: str) -> tp.Callable[..., tp.Any]:
            module = sys.modules[__name__]
            func_name = code.split("(")[0].split("def ")[-1]
            exec(code, globals())
            func = getattr(module, func_name)
            if is_callable(func, func_name):
                return func
            else:
                raise TypeError("The code must be function format.")

        # Aceエディタの初期値
        initial_code = """def your_problem():\n\t..."""
        # Aceエディタの設定
        editor_options = {
            "value": initial_code,
            "placeholder": "",
            "height": "300px",
            "language": "python",
            "theme": "ambiance",
            "keybinding": "vscode",
            "min_lines": 12,
            "max_lines": None,
            "font_size": 12,
            "tab_size": 4,
            "wrap": False,
            "show_gutter": True,
            "show_print_margin": False,
            "readonly": False,
            "annotations": None,
            "markers": None,
            "auto_update": False,
            "key": None,
        }

        code = codecs.decode(st_ace(**editor_options), "unicode_escape")
        # 実行ボタン
        if st.button("Run"):
            func = get_function_from_code(code)
            problem = func()
            st.latex(problem._repr_latex_()[2:-2])

    def on_select_result(self, session: Session) -> None:
        benchmark_id = session.state.selected_benchmark_id
        if benchmark_id:
            results_table = build_results_table(benchmark_id)
            options = [
                k
                for k, v in results_table.items()
                if isinstance(v[0], (int, float, np.int64, np.float64))
            ]

            with st.container():
                st.subheader("Scatter")
                cols = st.columns(3)
                with cols[0]:
                    xlabel = st.selectbox("X:", options=options, index=0)
                with cols[1]:
                    ylabel = st.selectbox("Y:", options=options, index=1)
                with cols[2]:
                    color = st.selectbox("Color:", options=options, index=2)

                fig = px.scatter(results_table, x=xlabel, y=ylabel, color=color)
                fig.update_layout(margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            with st.container():
                st.subheader("Parallel coordinates")
                selected_columns = st.multiselect(
                    "Columns", options=options, default=options[:5]
                )
                color = st.selectbox("Color", options=options)
                if color not in selected_columns:
                    selected_columns.append(color)
                fig = px.parallel_coordinates(
                    results_table[selected_columns],
                    color=color,
                )
                fig.update_layout(margin=dict(l=50))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            with st.container():
                st.subheader("Diff")
                cols = st.columns(2)
                with cols[0]:
                    r1_name = st.selectbox("Record 1", options=range(len(results_table)), index=0)
                with cols[1]:
                    r2_name = st.selectbox("Record 2", options=range(len(results_table)), index=1)
                
                with elements("diff"):
                    results = jb.load(benchmark_id)
                    r1 = results.data[1].data.iloc[r1_name]["sample_model_return[0]"]
                    r2 = results.data[1].data.iloc[r2_name]["sample_model_return[0]"]
                    editor.MonacoDiff(
                        original=r1.__repr__(),
                        modified=r2.__repr__(),
                        height=300,
                    )

            st.subheader("Table")
            gb = GridOptionsBuilder.from_dataframe(results_table)
            gb.configure_columns(
                ["benchmark_id", "experiment_id", "run_id"],
                width=100,
                cellStyle={"backgroundColor": "#f8f9fb"},
            )
            gridoptions = gb.build()
            AgGrid(results_table, gridOptions=gridoptions)

        st.subheader("Previous bencnmark")
        results_dir_series = pd.Series(glob.glob(f"{DEFAULT_RESULT_DIR}/*"))
        created_time_series = results_dir_series.apply(
            lambda x: str(
                datetime.datetime.fromtimestamp(pathlib.Path(x).stat().st_ctime)
            )
        )
        name_series = results_dir_series.apply(lambda x: pathlib.Path(x).name)
        path_table = pd.concat(
            [
                pd.Series([""] * len(results_dir_series)),
                created_time_series,
                name_series,
            ],
            axis=1,
        )
        path_table.columns = pd.Index(["", "timestamp", "benchmark_id"])

        gb = GridOptionsBuilder.from_dataframe(path_table)
        gb.configure_selection(use_checkbox=True)
        gb.configure_column("", width=42)
        gb.configure_column("timestamp", flex=1)
        gb.configure_column("benchmark_id", flex=3)
        grid_options = gb.build()
        grid_path_table = AgGrid(
            path_table,
            height=250,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )
        session.state.selected_benchmark_results = grid_path_table["selected_rows"]
        session.state.is_benchmark_results_loaded = st.button(
            "Load", key="load_benchmark_results"
        )


class State:
    def __init__(self) -> None:
        st.session_state["selected_page"] = "Instance data"
        st.session_state["instance_data_dir"] = InstanceDataDir()
        st.session_state["input_problem_name"] = ""
        st.session_state["uploaded_instance_data_name"] = ""
        st.session_state["selected_instance_data_name"] = None
        st.session_state["is_instance_data_loaded"] = False
        st.session_state["selected_instance_data_map"] = {}
        st.session_state["selected_figure_for_instance_data"] = None
        st.session_state["selected_benchmark_results"] = []
        st.session_state["is_benchmark_results_loaded"] = False

    @property
    def selected_page(self) -> str:
        return st.session_state["selected_page"]

    @selected_page.setter
    def selected_page(self, page: str) -> None:
        st.session_state["selected_page"] = page

    @property
    def instance_data_dir(self) -> InstanceDataDir:
        return st.session_state["instance_data_dir"]

    @property
    def input_problem_name(self) -> str:
        return st.session_state["input_problem_name"]

    @input_problem_name.setter
    def input_problem_name(self, problem_name: str) -> None:
        st.session_state["input_problem_name"] = problem_name

    @property
    def is_instance_data_loaded(self) -> bool:
        return st.session_state["is_instance_data_loaded"]

    @is_instance_data_loaded.setter
    def is_instance_data_loaded(self, b: bool) -> None:
        st.session_state["is_instance_data_loaded"] = b

    @property
    def uploaded_instance_data_name(self) -> str:
        return st.session_state["uploaded_instance_data_name"]

    @uploaded_instance_data_name.setter
    def uploaded_instance_data_name(self, instance_data_name: str) -> None:
        st.session_state["uploaded_instance_data_name"] = instance_data_name

    @property
    def selected_instance_data_name(self) -> str | None:
        return st.session_state["selected_instance_data_name"]

    @selected_instance_data_name.setter
    def selected_instance_data_name(self, instance_data_name: str | None) -> None:
        st.session_state["selected_instance_data_name"] = instance_data_name

    @property
    def selected_instance_data_map(self) -> dict[str, list[str]]:
        return st.session_state["selected_instance_data_map"]

    @selected_instance_data_map.setter
    def selected_instance_data_map(self, data_map: dict[str, list[str]]) -> None:
        st.session_state["selected_instance_data_map"] = data_map

    @property
    def selected_problem_names(self) -> list[str]:
        if "checked" in self.selected_instance_data_map:
            return [
                name
                for name in self.selected_instance_data_map["expanded"]
                if len(name.split("&")) == 1
            ]
        else:
            return []

    @property
    def selected_instance_data_files(self) -> list[str]:
        if "checked" in self.selected_instance_data_map:
            return self.selected_instance_data_map["checked"]
        else:
            return []

    @property
    def selected_figure_for_instance_data(self) -> str | None:
        return st.session_state["selected_figure_for_instance_data"]

    @selected_figure_for_instance_data.setter
    def selected_figure_for_instance_data(self, fig_type: str | None) -> None:
        st.session_state["selected_figure_for_instance_data"] = fig_type

    @property
    def selected_benchmark_results(self) -> list[dict[str, str | dict[str, str | int]]]:
        return st.session_state["selected_benchmark_results"]

    @selected_benchmark_results.setter
    def selected_benchmark_results(
        self, results: dict[str, str | dict[str, str | int]]
    ) -> None:
        st.session_state["selected_benchmark_results"] = results

    @property
    def selected_benchmark_id(self) -> str | None:
        if self.selected_benchmark_results:
            benchmark_id = self.selected_benchmark_results[0]["benchmark_id"]
            if isinstance(benchmark_id, str):
                return benchmark_id

    @property
    def is_benchmark_results_loaded(self) -> bool:
        return st.session_state["is_benchmark_results_loaded"]

    @is_benchmark_results_loaded.setter
    def is_benchmark_results_loaded(self, b: bool) -> None:
        st.session_state["is_benchmark_results_loaded"] = b


selected_rows = [
    {
        "_selectedRowNodeInfo": {"nodeRowIndex": 1, "nodeId": "1"},
        "": "",
        "timestamp": "2023-03-17 05:18:56.635438",
        "benchmark_id": "095d11b9-af66-432b-bc8d-94415ffe406b",
    }
]


class _Singleton(type):
    def __call__(cls, *args: tp.Any, **kwargs: tp.Any):
        if "_instance" not in st.session_state:
            st.session_state["_instance"] = super().__call__(*args, **kwargs)
        return st.session_state["_instance"]


class Session(metaclass=_Singleton):
    def __init__(self) -> None:
        self.state = State()
        self.paging_handler = PagingHandler()
        self.instance_data_handler = InstanceDataHandler()

    def display_page(self) -> None:
        self.paging_handler.on_select_page(self)

    def add_instance_data(self) -> None:
        self.instance_data_handler.on_add(self)

    def plot_instance_data(self) -> None:
        self.instance_data_handler.on_load(self)


@st.cache_data
def load_instance_data(files: list[str]) -> dict[str, dict[str, list[int | float]]]:
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


def plot_histogram_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
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


def plot_box_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(data))
    for i, (k, v) in enumerate(data.items()):
        fig.add_trace(go.Box(y=v, name=k), row=1, col=i + 1)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_violin_for_instance_data(data: dict[str, list[int | float]]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(data))
    for i, (k, v) in enumerate(data.items()):
        fig.add_trace(
            go.Violin(y=v, x0=k, name=k, box_visible=True, meanline_visible=True),
            row=1,
            col=i + 1,
        )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


@st.cache_data
def build_results_table(benchmark_id: str) -> pd.DataFrame:
    def expand_dict_series_in(table: pd.DataFrame) -> pd.DataFrame:
        expanded = pd.DataFrame()
        for c in table:
            sample = table[c][0]
            if isinstance(sample, dict):
                expanded = pd.concat(
                    [
                        expanded,
                        table.apply(lambda x: pd.Series(x[c]), axis=1).rename(
                            columns=lambda x: f"{c}[{x}]"
                        ),
                    ]
                )
                table.drop(columns=[c], inplace=True)
        return pd.concat([table, expanded], axis=1)

    results = jb.load(benchmark_id)
    e = jb.Evaluation()
    eval_table = e([results], opt_value=0).table.drop(columns=results.table.columns)

    params_table = results.params_table
    for c in ("model", "problem"):
        if c in params_table:
            params_table[c] = params_table[c].apply(lambda x: x.name)

    for c in ("feed_dict", "instance_data"):
        if c in params_table:
            params_table.drop(columns=[c], inplace=True)

    params_table = expand_dict_series_in(params_table)
    response_table = expand_dict_series_in(results.response_table)
    return pd.concat([params_table, response_table, eval_table], axis=1).reset_index()


# def sidebar_for_instance_data() -> None:
#    with st.sidebar:
#        with st.expander("Instance data"):
#            session.state.selected_instance_data_map = tree_select(
#                session.state.instance_data_dir.nodes,
#                check_model="leaf",
#                only_leaf_checkboxes=True,
#            )
#            with st.expander("Register new instance data"):
#                with st.form("new_instance_data"):
#                    session.state.input_problem_name = st.text_input(
#                        "Input problem name"
#                    )
#                    byte_stream = st.file_uploader(
#                        "Upload your instance data", type=["json"]
#                    )
#                    if byte_stream:
#                        session.state.uploaded_instance_data_name = byte_stream.name
#                        ins_d = rapidjson.loads(byte_stream.getvalue())
#                    if st.form_submit_button("Submit"):
#                        if byte_stream:
#                            session.add_instance_data()
#                            st.experimental_rerun()
#            if st.button("Load"):
#                session.state.is_instance_data_loaded = True
#                st.experimental_rerun()


# def sidebar_for_problem() -> None:
#    with st.sidebar:
#        pass


session = Session()


def main():
    st.title("JB Board")

    tab_names = ["Instance data", "Problem", "Solver", "Analysis"]
    tab_map = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

    with tab_map["Instance data"]:
        session.state.selected_page = "Instance data"
        session.display_page()

    with tab_map["Problem"]:
        session.state.selected_page = "Problem"
        session.display_page()

    with tab_map["Solver"]:
        session.state.selected_page = "Solver"
        session.display_page()

    with tab_map["Analysis"]:
        session.state.selected_page = "Analysis"
        session.display_page()

    # session = Session()
    # st.info(session.state.selected_instance_data_files)


if __name__ == "__main__":

    main()

# results = jb.load("test")
#
# mplot = MetricsPlot(results)
# fig = mplot.parallelplot_experiment()
# plotly_events(fig, select_event=True, hover_event=True)

# # データのロード（例として、seabornのサンプルデータを使用）
# @st.cache
# def load_data():
#     data = pd.read_csv(
#         "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
#     )
#     return data
#
#
# data = load_data()
#
# # タイトルとサブタイトル
# st.title("Streamlitダッシュボード")
# st.subheader("AWS QuickSight風")
#
# # フィルタリング用のサイドバー
# st.sidebar.header("フィルター設定")
# min_sepal_length = st.sidebar.slider(
#     "がく片の長さ（最小値）",
#     float(data["sepal_length"].min()),
#     float(data["sepal_length"].max()),
#     float(data["sepal_length"].min()),
#     0.1,
# )
# selected_species = st.sidebar.selectbox("花の種類を選択", sorted(data["species"].unique()))
#
# # データのフィルタリング
# filtered_data = data[
#     (data["sepal_length"] >= min_sepal_length) & (data["species"] == selected_species)
# ]
#
# # フィルタリングされたデータを表示
# st.subheader(f"選択された花の種類: {selected_species}")
# st.write(filtered_data)
#
# # データの可視化
# chart = (
#     alt.Chart(filtered_data)
#     .mark_circle(size=60)
#     .encode(
#         x="sepal_length",
#         y="sepal_width",
#         color="species",
#         tooltip=[
#             "species",
#             "sepal_length",
#             "sepal_width",
#             "petal_length",
#             "petal_width",
#         ],
#     )
#     .interactive()
# )
#
# st.altair_chart(chart, use_container_width=True)


# class InstanceDataFolderTree:
#     jijbench_default_problem_names: list[str] = [
#         "BinPacking",
#         "Knapsack",
#         "TSP",
#         "TSPTW",
#         "NurseScheduling",
#     ]
#
#     def __init__(self) -> None:
#         default_datasets = [
#             getattr(datasets, problem_name)()
#             for problem_name in self.jijbench_default_problem_names
#         ]
#         self._nodes = [
#             {
#                 "label": dataset.problem_name,
#                 "value": dataset.problem_name,
#                 "children": [
#                     {
#                         "label": size,
#                         "value": f"{dataset.problem_name}_{size}",
#                         "children": [
#                             {"label": ins_d_name, "value": ins_d_name}
#                             for ins_d_name in dataset.instance_names(size)
#                         ],
#                     }
#                     for size in ["small", "medium", "large"]
#                 ],
#             }
#             for dataset in default_datasets
#         ]
#
#     @property
#     def nodes(self) -> list[dict[str, tp.Any]]:
#         return self._nodes
#
#     def add_instance_data(self, problem_name: str, data: dict[str, tp.Any]) -> None:
#         pass
#
#     def create_folder_tree(self):
#         pass
