from __future__ import annotations

import codecs
import datetime
import glob
import pathlib
import re
import sys
import typing as tp

import jijmodeling as jm
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import rapidjson
import streamlit as st
from plotly.colors import n_colors
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridUpdateMode
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
            self.on_select_analysis(session)

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

        ph_plot = st.empty()

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
                    with ph_plot.container():
                        session.plot_instance_data()

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

    def on_select_analysis(self, session: Session) -> None:
        """
        Display the benchmark results and analysis options.

        Args:
            session (Session): The current session.
        """

        components = {
            "id_table": st.empty(),
            "result_table": st.empty(),
            "evaluation_table": st.empty(),
            "sampleset_analysis": st.empty(),
            "scatter": st.empty(),
            "parallel_coordinates": st.empty(),
        }

        with components["id_table"].container():
            st.subheader("Experiment history")
            id_table = jb.get_id_table(savedir=session.state.logdir)
            id_table = (
                id_table.sort_values("timestamp", ascending=False)
                .replace("*", "⭐️")
                .drop(columns=["savedir"])
            )
            id_table.insert(0, "", [""] * len(id_table))

            gob = GridOptionsBuilder.from_dataframe(id_table)
            gob.configure_selection(use_checkbox=True, selection_mode="multiple")
            gob.configure_column("star", cellStyle={"textAlign": "center"})
            grid_options = gob.build()
            grid_id_table = AgGrid(
                id_table,
                height=250,
                gridOptions=grid_options,
                grid_update_mode=GridUpdateMode.VALUE_CHANGED,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            )
            session.state.selected_benchmark_results = grid_id_table["selected_rows"]
            is_benchmark_results_loaded = st.button(
                "Load", key="load_benchmark_results"
            )
            if is_benchmark_results_loaded:
                session.state.num_experiment_loaded += 1

        table_cols = st.columns(2)
        if session.state.num_experiment_loaded:
            benchmark_ids = session.state.selected_benchmark_ids
            params_table = _get_params_table(benchmark_ids, session.state.logdir)

            with components["result_table"].container():
                st.subheader("Table")
                response_table = _get_response_table(
                    benchmark_ids, session.state.logdir
                )
                table = pd.concat([params_table, response_table], axis=1)
                gob = GridOptionsBuilder.from_dataframe(table)
                gob.configure_columns(
                    params_table.columns,
                    cellStyle={"backgroundColor": "#f8f9fb"},
                    pinned="left",
                )
                grid_options = gob.build()
                AgGrid(
                    table,
                    height=500,
                    gridOptions=grid_options,
                    grid_update_mode=GridUpdateMode.VALUE_CHANGED,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                )
                st.markdown("<hr>", unsafe_allow_html=True)

            with components["evaluation_table"].container():
                st.subheader("Evaluation")
                cols = st.columns(8)
                with cols[0]:
                    opt_value = st.text_input("opt value", value=None)
                    if _is_number_str(opt_value):
                        opt_value = float(opt_value)
                    else:
                        opt_value = None
                with cols[1]:
                    pr = st.text_input(
                        "pr",
                        value=0.99,
                    )
                    if _is_number_str(pr):
                        pr = float(pr)
                evaluation_table = _get_evaluation_table(
                    benchmark_ids, session.state.logdir
                )
                table = pd.concat([params_table, evaluation_table], axis=1)
                gob = GridOptionsBuilder.from_dataframe(table)
                gob.configure_columns(
                    params_table.columns,
                    cellStyle={"backgroundColor": "#f8f9fb"},
                    pinned="left",
                )
                grid_options = gob.build()
                AgGrid(
                    table,
                    height=500,
                    gridOptions=grid_options,
                    grid_update_mode=GridUpdateMode.VALUE_CHANGED,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                )
                st.markdown("<hr>", unsafe_allow_html=True)

            # with ph_scatter.container():
            #     st.subheader("Scatter")
            #     cols = st.columns(3)
            #     with cols[0]:
            #         xlabel = st.selectbox("X:", options=options, index=0)
            #     with cols[1]:
            #         ylabel = st.selectbox("Y:", options=options, index=1)
            #     with cols[2]:
            #         color = st.selectbox("Color:", options=options, index=2)

            #     fig = px.scatter(results_table, x=xlabel, y=ylabel, color=color)
            #     fig.update_layout(margin=dict(t=10))
            #     st.plotly_chart(fig, use_container_width=True)

            #     st.markdown("<hr>", unsafe_allow_html=True)

            # with ph_parallel_coordinates.container():
            #     st.subheader("Parallel coordinates")
            #     selected_columns = st.multiselect(
            # "Columns", options=options, default=options[:5]
            #     )
            #     color = st.selectbox("Color", options=options)
            #     if color not in selected_columns:
            #         selected_columns.append(color)
            #     fig = px.parallel_coordinates(
            #         results_table[selected_columns],
            #         color=color,
            #     )
            #     fig.update_layout(margin=dict(l=50))
            #     st.plotly_chart(fig, use_container_width=True)

            #     st.markdown("<hr>", unsafe_allow_html=True)

            with components["sampleset_analysis"].container():
                st.subheader("SampleSet Analysis")
                st.subheader("Constraint Violations")

                violations_analysis_cols = st.columns(2)
                plot_settings_cols = st.columns(2)
                plot_components = {}

                with violations_analysis_cols[0]:
                    # bar
                    plot_components["bar"] = st.empty()
                    plot_components["settings_bar"] = st.empty()
                    # st.markdown("<hr>", unsafe_allow_html=True)

                    with plot_components["settings_bar"]:
                        agg_options = ["min", "max", "mean", "std"]
                        groupby_options: list[str] = params_table.columns.tolist()

                        with plot_settings_cols[0]:
                            cols = st.columns(2)
                            with cols[0]:
                                agg = (
                                    st.selectbox(
                                        "Select a function for aggregating violations:",
                                        options=agg_options,
                                        index=0,
                                    )
                                    or "min"
                                )
                            with cols[1]:
                                groupby = (
                                    st.selectbox(
                                        "Group by:", options=groupby_options, index=0
                                    )
                                    or groupby_options[0]
                                )

                    table = pd.concat([params_table, evaluation_table], axis=1)
                    unselected_options = [c for c in params_table if c != groupby]
                    violations_table = table.filter(
                        regex="|".join(groupby_options + [f"violations_{agg}"])
                    )

                    violations_table = violations_table.rename(
                        columns={
                            c: c.replace(f"_violations_{agg}", "")
                            for c in violations_table.columns
                            if c.endswith(f"violations_{agg}")
                        }
                    )
                    violations_table = violations_table.astype(
                        {c: str for c in groupby_options}
                    )
                    stacked = violations_table.set_index(
                        params_table.columns.tolist()
                    ).stack()
                    stacked.name = f"violations_{agg}"
                    stacked = stacked.reset_index(unselected_options)

                    fig = px.bar(
                        stacked,
                        x=stacked.index.get_level_values(-1),
                        y=stacked[f"violations_{agg}"],
                        color=stacked.index.get_level_values(groupby),
                        barmode="group",
                        hover_data=unselected_options,
                        color_discrete_sequence=px.colors.sequential.Jet,
                    )

                    with plot_components["bar"].container():
                        st.plotly_chart(fig, use_container_width=True)

                with violations_analysis_cols[1]:
                    # line
                    plot_components["line"] = st.empty()
                    plot_components["settings_line"] = st.empty()

                    with plot_components["settings_line"]:
                        array_options: list[str] = [
                            c.split("_mean")[0]
                            for c in evaluation_table.filter(regex="_mean").columns
                        ]
                        metric_options = [
                            "success_probability",
                            "feasible_rate",
                            "residual_energy",
                            "TTS[optimal]",
                            "TTS[feasible]",
                            "TTS[derived]",
                        ]
                        with plot_settings_cols[1]:
                            cols = st.columns(3)
                            with cols[0]:
                                x = (
                                    st.selectbox(
                                        "X:", options=groupby_options, index=0, key="x"
                                    )
                                    or groupby_options[0]
                                )
                            with cols[1]:
                                base_y = (
                                    st.selectbox(
                                        "Y:",
                                        options=array_options + metric_options,
                                        index=0,
                                        key="y",
                                    )
                                    or array_options[0]
                                )
                                if base_y in array_options:
                                    y = base_y + "_mean"
                                else:
                                    y = base_y
                            with cols[2]:
                                unselected_options = [c for c in params_table if c != x]
                                groupby = st.selectbox(
                                    "Group by:",
                                    options=unselected_options,
                                    index=0,
                                    key="groupby",
                                )

                        unselected_options = [
                            c for c in params_table if c not in (x, groupby)
                        ]
                        with plot_settings_cols[1]:
                            cols = st.columns(len(unselected_options))
                            index: list[str] = []
                            for col, option in zip(cols, unselected_options):
                                with col:
                                    unique_values = (
                                        params_table[option].unique().tolist()
                                    )
                                    index.append(
                                        st.selectbox(
                                            f"{option}:",
                                            options=unique_values,
                                            index=0,
                                        )
                                        or unique_values[0]
                                    )

                    index_x = tuple(index + [x])
                    index_y = tuple(index + [y])
                    index_lower_y = tuple(index + [f"{base_y}_std"])

                    fig = go.Figure()
                    for i, (name, group) in enumerate(table.groupby(groupby)):
                        line_color = px.colors.sequential.Jet[
                            i % len(px.colors.sequential.Jet)
                        ]
                        fill_color = px.colors.sequential.Jet[
                            i % len(px.colors.sequential.Jet)
                        ]

                        stacked = (
                            group.set_index(unselected_options).stack().sort_index()
                        )
                        lower = (
                            stacked.loc[index_y].values
                            - stacked.loc[index_lower_y].values
                        )
                        upper = (
                            stacked.loc[index_y].values
                            + stacked.loc[index_lower_y].values
                        )

                        fig.add_traces(
                            [
                                go.Scatter(
                                    x=stacked.loc[index_x].values,
                                    y=stacked.loc[index_y].values,
                                    name=name,
                                    mode="markers+lines",
                                    line=dict(color=line_color),
                                ),
                                go.Scatter(
                                    x=stacked.loc[index_x].values,
                                    y=upper,
                                    name=name,
                                    showlegend=False,
                                    mode="lines",
                                    fillcolor=_rgb_to_rgba(fill_color, 0.2),
                                    line=dict(color=_rgb_to_rgba(fill_color, 0.0)),
                                ),
                                go.Scatter(
                                    x=stacked.loc[index_x].values,
                                    y=lower,
                                    fill="tonexty",
                                    name=name,
                                    showlegend=False,
                                    mode="lines",
                                    fillcolor=_rgb_to_rgba(fill_color, 0.2),
                                    line=dict(color=_rgb_to_rgba(fill_color, 0.0)),
                                ),
                            ]
                        )
                    fig.update_layout(xaxis=dict(title=x), yaxis=dict(title=base_y))
                    fig.update_coloraxes(colorscale="Jet")

                    with plot_components["line"].container():
                        st.plotly_chart(fig, use_container_width=True)

                for col in st.columns(2):
                    with col:
                        st.markdown("<hr>", unsafe_allow_html=True)

            # tmp
            # concat: jb.functions.Concat[jb.Experiment] = jb.functions.Concat()
            # results = concat(
            #     [
            #         jb.load(benchmark_id, savedir=session.state.logdir)
            #         for benchmark_id in benchmark_ids
            #     ]
            # )
            # _, t = results.data
            # heatmap = []
            # for name, group in t.view().groupby("solver_name"):
            #     for sampleset in t.data.loc[group.index, "solver_output[0]"]:
            #         expr_values = sampleset.evaluation.constraint_expr_values[-1]
            #         expr_values = {str(k): v for k, v in expr_values.items()}
            #         x = expr_values["assign"]
            #         keys = [str(k) for k in x.keys()]
            #         values = list(x.values())
            #         heatmap.append(values)
            # heatmap = np.array(heatmap).T
            # fig = px.imshow(heatmap, aspect="auto", color_continuous_scale="Viridis")
            # st.plotly_chart(fig, use_container_width=True)
            # st.markdown("<hr>", unsafe_allow_html=True)

            # with ph_sampleset_diff.container():
            #    st.subheader("Diff")
            #    cols = st.columns(2)
            #    with cols[0]:
            #        r1_name = st.selectbox(
            #            "Record 1", options=range(len(results_table)), index=0
            #        )
            #    with cols[1]:
            #        r2_name = st.selectbox(
            #            "Record 2", options=range(len(results_table)), index=1
            #        )

            #    with elements("diff"):
            #        results = jb.load(benchmark_id, savedir=session.state.logdir)
            #        r1 = results.data[1].data.iloc[r1_name]["solver_output[0]"]
            #        r2 = results.data[1].data.iloc[r2_name]["solver_output[0]"]
            #        editor.MonacoDiff(
            #            original="\n".join(
            #                r1.__repr__()[i : i + 100]
            #                for i in range(0, len(r1.__repr__()), 100)
            #            ),
            #            modified="\n".join(
            #                r2.__repr__()[i : i + 100]
            #                for i in range(0, len(r2.__repr__()), 100)
            #            ),
            #            height=300,
            #        )


@st.cache_data
def _get_params_table(benchmark_ids: list[str], savedir: pathlib.Path) -> pd.DataFrame:
    table = jb.load(benchmark_ids[0], savedir=savedir).params_table
    expected_problem_columns = [c for c in table if isinstance(table[c][0], jm.Problem)]
    for benchmark_id in benchmark_ids[1:]:
        params_table = jb.load(benchmark_id, savedir=savedir).params_table
        problem_columns = [
            c for c in params_table if isinstance(params_table[c][0], jm.Problem)
        ]
        for c, expected in zip(problem_columns, expected_problem_columns):
            params_table = params_table.rename(columns={c: expected})

        table = pd.concat([table, params_table])

    for c in ("feed_dict", "instance_data"):
        droped_columns = table.filter(regex=c).columns
        table = table.drop(columns=droped_columns)

    for c in expected_problem_columns:
        table[c] = table[c].apply(lambda x: x.name)
    return table


@st.cache_data
def _get_response_table(
    benchmark_ids: list[str], savedir: pathlib.Path
) -> pd.DataFrame:
    table = pd.DataFrame()
    for benchmark_id in benchmark_ids:
        results = jb.load(benchmark_id, savedir=savedir)
        response_table = jb.Table._expand_dict_in(results.response_table)
        table = pd.concat([table, response_table])
    return table


@st.cache_data
def _get_evaluation_table(
    benchmark_ids: list[str],
    savedir: pathlib.Path,
    opt_value: float | None = None,
    pr: float = 0.99,
) -> pd.DataFrame:
    table = pd.DataFrame()
    for benchmark_id in benchmark_ids:
        results = jb.load(benchmark_id, savedir=savedir)
        e = jb.Evaluation()
        evaluation_table = e([results], opt_value=opt_value, pr=pr).table.drop(
            columns=results.table.columns,
        )
        table = pd.concat([table, evaluation_table])
    return table


def _rgb_to_rgba(rgb_str: str, alpha: float) -> str:
    rgba = rgb_str.replace("rgb", "rgba")
    rgba = rgba.split(")")
    rgba[-1] = f"{alpha}"
    return ",".join(rgba) + ")"


def _is_number_str(s: str) -> bool:
    pattern = r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
    return bool(re.match(pattern, s))
