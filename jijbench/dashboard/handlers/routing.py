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
from streamlit.delta_generator import DeltaGenerator
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
            "problem": st.empty(),
            "metric_plot": st.empty(),
            "sampleset_analysis": st.empty(),
            "scatter": st.empty(),
            "parallel_coordinates": st.empty(),
        }
        plot_components: dict[str, DeltaGenerator] = {}

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

        if session.state.num_experiment_loaded:
            benchmark_ids = session.state.selected_benchmark_ids
            params_table = _get_params_table(benchmark_ids, session.state.logdir)

            with components["result_table"].container():
                st.subheader("Table")
                response_table = _get_response_table(
                    benchmark_ids, session.state.logdir
                )
                table = pd.concat([params_table, response_table], axis=1)
                st.dataframe(table)
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

            with components["problem"].container():
                st.subheader("Problems")

                def get_unique_problems(x: pd.Series) -> pd.Series:
                    problem_names = x.apply(lambda x: x.name)
                    problem_names.name = f"{x.name}_name"
                    return pd.concat([x, problem_names], axis=1).drop_duplicates(
                        f"{x.name}_name"
                    )[x.name]

                problem_table = _get_problem_table(benchmark_ids, session.state.logdir)
                problems = np.unique(
                    [
                        p
                        for problems in problem_table.apply(get_unique_problems).values
                        for p in problems
                    ]
                ).tolist()
                for problem in problems:
                    with st.expander(f"{problem.name}"):
                        st.latex(problem._repr_latex_()[2:-2])
                st.markdown("<hr>", unsafe_allow_html=True)

            with components["metric_plot"].container():
                st.subheader("SampleSet Analysis")
                # line
                st.subheader("Metric")
                plot_components["line"] = st.empty()
                plot_components["settings_line"] = st.empty()

                with plot_components["settings_line"].container():
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
                    groupby_options: list[str] = params_table.columns.tolist()

                    cols = st.columns(2)
                    with cols[0]:
                        groupby = st.multiselect(
                            "Group by:",
                            options=groupby_options,
                            default=groupby_options[0],
                            key="groupby",
                        )
                        x_labels = [o for o in groupby_options if o not in groupby]
                    with cols[1]:
                        base_y_labels = (
                            st.multiselect(
                                "Metrics:",
                                options=array_options + metric_options,
                                default=array_options[0],
                                key="metric",
                            )
                            or array_options[0]
                        )

                figs = []
                for base_y_label in base_y_labels:
                    if base_y_label in array_options:
                        if "violations" in base_y_label:
                            y_label = base_y_label + "_min"
                        else:
                            y_label = base_y_label + "_mean"
                    else:
                        y_label = base_y_label

                    fig = go.Figure()
                    for i, (name, group) in enumerate(params_table.groupby(groupby)):
                        if isinstance(name, str):
                            name = [name]
                        name = [str(n) for n in name]

                        line_color = px.colors.sequential.Jet[
                            i % len(px.colors.sequential.Jet)
                        ]
                        fill_color = px.colors.sequential.Jet[
                            i % len(px.colors.sequential.Jet)
                        ]

                        x = (
                            group[x_labels]
                            .astype(str)
                            .apply(lambda x: ", ".join(x), axis=1)
                        )
                        y = evaluation_table.loc[group.index, y_label]
                        upper = (
                            y + evaluation_table.loc[group.index, f"{base_y_label}_std"]
                        )
                        lower = (
                            y - evaluation_table.loc[group.index, f"{base_y_label}_std"]
                        )

                        if "violations" in base_y_label:
                            fig.add_traces(
                                [
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        name=", ".join(name),
                                        mode="markers+lines",
                                        line=dict(color=line_color),
                                    )
                                ]
                            )
                        else:
                            fig.add_traces(
                                [
                                    go.Scatter(
                                        x=x,
                                        y=y,
                                        name=", ".join(name),
                                        mode="markers+lines",
                                        line=dict(color=line_color),
                                    ),
                                    go.Scatter(
                                        x=x,
                                        y=upper,
                                        name=", ".join(name),
                                        showlegend=False,
                                        mode="lines",
                                        fillcolor=_rgb_to_rgba(fill_color, 0.2),
                                        line=dict(color=_rgb_to_rgba(fill_color, 0.0)),
                                    ),
                                    go.Scatter(
                                        x=x,
                                        y=lower,
                                        fill="tonexty",
                                        name=", ".join(name),
                                        showlegend=False,
                                        mode="lines",
                                        fillcolor=_rgb_to_rgba(fill_color, 0.2),
                                        line=dict(color=_rgb_to_rgba(fill_color, 0.0)),
                                    ),
                                ]
                            )
                        fig.update_layout(
                            title=f"{y_label.replace('_mean', '').capitalize()} grouped by {groupby}",
                            xaxis=dict(
                                automargin=True,
                                title=", ".join(x_labels),
                                type="category",
                                showgrid=True,
                            ),
                            yaxis=dict(
                                automargin=True, title=base_y_label, showgrid=True
                            ),
                            coloraxis=dict(colorscale="jet"),
                            legend=dict(x=1, y=1, xanchor="left", yanchor="top"),
                        )
                    figs.append(fig)

                with plot_components["line"].container():
                    for fig in figs:
                        st.plotly_chart(fig, use_container_width=True)

            # with components["scatter"].container():
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

            # parallel_coordinates
            with components["parallel_coordinates"].container():
                categorical_columns = params_table.select_dtypes(
                    include="object"
                ).columns.tolist()

                for name, _ in params_table.groupby(categorical_columns):
                    plot_components[
                        f"parallel_coordinates_{'_'.join(name)}"
                    ] = st.empty()

                metrics = ["feasible_rate"]
                filterd_evaluation_table = pd.concat(
                    [evaluation_table.filter(regex="_min"), evaluation_table[metrics]],
                    axis=1,
                ).rename(columns=lambda x: x.replace("_min", ""))
                cols = st.columns(4)
                with cols[0]:
                    color_options = filterd_evaluation_table.columns
                    color = st.selectbox(
                        "Color:",
                        options=color_options,
                        index=color_options.get_loc("objective"),
                    )

                for name, group in params_table.groupby(categorical_columns):
                    parcoords_table = pd.concat(
                        [
                            group,
                            filterd_evaluation_table.loc[group.index],
                        ],
                        axis=1,
                    )
                    max_column_length = max([len(str(c)) for c in parcoords_table])

                    fig = px.parallel_coordinates(
                        parcoords_table.reset_index(drop=True),
                        color=color,
                        color_continuous_scale="tealrose",
                    )
                    fig.update_traces(labelangle=30)
                    fig.update_coloraxes(colorbar=dict(titleside="right"))
                    fig.update_layout(
                        margin=dict(l=100, t=3.85 * max_column_length + 25),
                        title_text=", ".join(name),
                        title_y=0.98,
                    )

                    with plot_components[f"parallel_coordinates_{'_'.join(name)}"]:
                        st.plotly_chart(fig, use_container_width=True)

                # st.markdown("<hr>", unsafe_allow_html=True)

            with components["sampleset_analysis"].container():
                st.subheader("Constraint violations")

                # violations_analysis_cols = st.columns(2)

                # bar
                plot_components["bar"] = st.empty()
                plot_components["settings_bar"] = st.empty()
                # st.markdown("<hr>", unsafe_allow_html=True)
                with plot_components["settings_bar"]:
                    agg_options = ["min", "max", "mean", "std"]
                    groupby_options: list[str] = params_table.columns.tolist()
                    cols = st.columns(4)
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
                            st.selectbox("Group by:", options=groupby_options, index=0)
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

            constraint_heatmap = st.columns(2)
            # constraint violations
            with constraint_heatmap[0]:
                numeric_columns = params_table.select_dtypes("number").columns
                categorical_columns = params_table.select_dtypes("object").columns
                st.subheader("Heatmap for constraint violations")
                for name, _ in params_table.groupby(categorical_columns.tolist()):
                    plot_components[f"heatmap_violations_{'_'.join(name)}"] = st.empty()

                violations_table = evaluation_table.filter(regex="violations_min")

                cols = st.columns(2)
                with cols[0].container():
                    x_labels = st.multiselect(
                        "x:",
                        options=numeric_columns,
                        default=numeric_columns[0],
                        key="constraint_violations",
                    )
                for name, group in params_table.groupby(categorical_columns.tolist()):
                    fig = px.imshow(
                        violations_table.loc[group.index].T,
                        x=group[x_labels]
                        .astype(str)
                        .apply(lambda x: ", ".join(x), axis=1),
                        y=violations_table.loc[group.index]
                        .rename(columns=lambda x: x.replace("_violations_min", ""))
                        .columns,
                        labels=dict(x=f"{', '.join(x_labels)}", y="Constraint name"),
                        color_continuous_scale="deep_r",
                        aspect="auto",
                    )
                    fig.update_layout(
                        title_text=", ".join(name),
                        title_x=0,
                        xaxis=dict(automargin=True, type="category"),
                        yaxis=dict(automargin=True),
                    )
                    with plot_components[
                        f"heatmap_violations_{'_'.join(name)}"
                    ].container():
                        st.plotly_chart(fig, use_container_width=True)
                # st.markdown("<hr>", unsafe_allow_html=True)

            # constraint expr values
            with constraint_heatmap[1]:
                st.subheader("Heatmap for constraint expression values")
                for name, _ in params_table.groupby(categorical_columns.tolist()):
                    plot_components[
                        f"heatmap_expression_values_{'_'.join(name)}"
                    ] = st.empty()

                def expand(x: jm.SampleSet) -> pd.Series:
                    violations = x.evaluation.constraint_violations or {}
                    violations_argmin_map = {
                        k: np.argmin(v) for k, v in violations.items()
                    }

                    expr_values_map = {k: {} for k in violations}
                    for const_name in violations:
                        argmin = violations_argmin_map[const_name]
                        expr_values_map[const_name].update(
                            x.evaluation.constraint_expr_values[argmin][const_name]
                        )

                    return expr_values_map

                concat: jb.functions.Concat[jb.Experiment] = jb.functions.Concat()
                results = concat(
                    [
                        jb.load(benchmark_id, savedir=session.state.logdir)
                        for benchmark_id in benchmark_ids
                    ]
                )
                _, t = results.data
                sampleset_column = [
                    c for c in t.data if isinstance(t.data[c][0], jb.SampleSet)
                ][0]
                sampleset_table = t.data.filter([sampleset_column]).applymap(
                    lambda x: x.data
                )
                sampleset_table = pd.concat([params_table, sampleset_table], axis=1)

                sampleset_table = pd.concat(
                    [params_table, sampleset_table[sampleset_column].apply(expand)],
                    axis=1,
                )

                cols = st.columns(2)
                with cols[0].container():
                    const_name = st.selectbox(
                        "Select constraint:",
                        options=sampleset_table[sampleset_column][0].keys(),
                        index=0,
                    )
                with cols[1].container():
                    x_labels = st.multiselect(
                        "x:",
                        options=numeric_columns,
                        default=numeric_columns[0],
                        key="constraint_expr_values",
                    )

                for name, group in sampleset_table.groupby(
                    categorical_columns.tolist()
                ):
                    index = list(
                        map(
                            lambda x: str(x),
                            group[sampleset_column][0][const_name].keys(),
                        )
                    )
                    x = (
                        group[x_labels]
                        .astype(str)
                        .apply(lambda x: ", ".join(x), axis=1)
                    )
                    expr_values = np.array(
                        group[sampleset_column]
                        .apply(lambda x: list(x[const_name].values()))
                        .tolist(),
                        dtype=np.float64,
                    ).T

                    fig = px.imshow(
                        expr_values,
                        x=x,
                        y=index,
                        color_continuous_scale="deep_r",
                        labels=dict(x=f"{', '.join(x_labels)}", y="Index"),
                        aspect="auto",
                    )
                    fig.update_layout(
                        title_text=", ".join(name),
                        title_x=0,
                        xaxis=dict(automargin=True, type="category"),
                        yaxis=dict(automargin=True),
                        coloraxis=dict(
                            colorbar=dict(title=const_name, titleside="right")
                        ),
                    )

                    with plot_components[
                        f"heatmap_expression_values_{'_'.join(name)}"
                    ].container():
                        st.plotly_chart(fig, use_container_width=True)
                # st.markdown("<hr>", unsafe_allow_html=True)

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
def _get_problem_table(benchmark_ids: list[str], savedir: pathlib.Path) -> pd.DataFrame:
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
    return table[expected_problem_columns]


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
