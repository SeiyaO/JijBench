from __future__ import annotations

import typing as tp

import streamlit as st

from jijbench.dashboard.handlers.instance_data import (
    InstanceDataDir,
    InstanceDataHandler,
)
from jijbench.dashboard.handlers.routing import RoutingHandler


class _Singleton(type):
    def __call__(cls, *args: tp.Any, **kwargs: tp.Any):
        if "_instance" not in st.session_state:
            st.session_state["_instance"] = super().__call__(*args, **kwargs)
        return st.session_state["_instance"]


class Session(metaclass=_Singleton):
    def __init__(self) -> None:
        self.state = State()
        self.routing_handler = RoutingHandler()
        self.instance_data_handler = InstanceDataHandler()

    def display_page(self) -> None:
        self.routing_handler.on_select_page(self)

    def add_instance_data(self) -> None:
        self.instance_data_handler.on_add(self)

    def plot_instance_data(self) -> None:
        self.instance_data_handler.on_load(self)


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
