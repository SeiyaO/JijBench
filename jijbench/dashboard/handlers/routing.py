from __future__ import annotations

import codecs
import datetime
import glob
import pathlib
import sys
import typing as tp

import numpy as np
import pandas as pd
import plotly.express as px
import rapidjson
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_ace import st_ace
from streamlit_elements import editor, elements
from streamlit_tree_select import tree_select
from typeguard import check_type
from typing_extensions import TypeGuard

import jijbench as jb

if tp.TYPE_CHECKING:
    from jijbench.dashboard.session import Session


class RoutingHandler:
    """
    This class provides methods to handle the navigation between different pages of the application,
    such as instance data selection, solver configuration, problem definition, and result analysis.
    """

    def on_select_page(self, session: Session) -> None:
        """
        Handle the navigation to the selected page.

        Args:
            session (Session): The current session.
        """
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
        """
        Display the instance data selection and visualization options.

        Args:
            session (Session): The current session.
        """
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
                    session.state.instance_data_dir_tree.nodes,
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
        """
        Display the solver selection and configuration options.

        Args:
            session (Session): The current session.
        """
        st.info("Coming soon...")

    def on_select_problem(self, session: Session) -> None:
        """
        Display the problem definition and visualization options.

        Args:
            session (Session): The current session.
        """

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

        if st.button("Run"):
            st.info("Coming soon...")
            # func = get_function_from_code(code)
            # problem = func()
            # st.latex(problem._repr_latex_()[2:-2])

    def on_select_result(self, session: Session) -> None:
        """
        Display the benchmark results and analysis options.

        Args:
            session (Session): The current session.
        """
        benchmark_id = session.state.selected_benchmark_id
        if benchmark_id:
            results_table = _get_results_table(benchmark_id, session.state.logdir)
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
                    r1_name = st.selectbox(
                        "Record 1", options=range(len(results_table)), index=0
                    )
                with cols[1]:
                    r2_name = st.selectbox(
                        "Record 2", options=range(len(results_table)), index=1
                    )

                with elements("diff"):
                    results = jb.load(benchmark_id, savedir=session.state.logdir)
                    r1 = results.data[1].data.iloc[r1_name]["sample_model_return[0]"]
                    r2 = results.data[1].data.iloc[r2_name]["sample_model_return[0]"]
                    editor.MonacoDiff(
                        original="\n".join(
                            r1.__repr__()[i : i + 100]
                            for i in range(0, len(r1.__repr__()), 100)
                        ),
                        modified="\n".join(
                            r2.__repr__()[i : i + 100]
                            for i in range(0, len(r2.__repr__()), 100)
                        ),
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
        results_dir_series = pd.Series(glob.glob(f"{session.state.logdir}/*"))
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


@st.cache_data
def _get_results_table(benchmark_id: str, savedir: pathlib.Path) -> pd.DataFrame:
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

    results = jb.load(benchmark_id, savedir=savedir)
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