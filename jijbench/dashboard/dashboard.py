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





selected_rows = [
    {
        "_selectedRowNodeInfo": {"nodeRowIndex": 1, "nodeId": "1"},
        "": "",
        "timestamp": "2023-03-17 05:18:56.635438",
        "benchmark_id": "095d11b9-af66-432b-bc8d-94415ffe406b",
    }
]





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
