"""
This script provides a dashboard application using Streamlit to visualize and analyze experiment for optimization problem.
The dashboard consists of four tabs: "Instance data", "Problem", "Solver", and "Analysis",
allowing users to interact with and explore the benchmark data in various ways.

Usage:
To run the dashboard application, execute this script using a Python interpreter. Make sure that Streamlit
and other required packages are installed in your Python environment.

Attributes:
logdir (str): An optional environment variable specifying the directory containing the benchmark data.
If not provided, the default directory defined in DEFAULT_RESULT_DIR is used.

Functions:
run(): Initializes the dashboard application and sets up the tabs for user interaction.
"""

from __future__ import annotations

import os
import pathlib

import streamlit as st

from jijbench.consts.default import DEFAULT_RESULT_DIR
from jijbench.dashboard.session import Session

st.set_page_config(layout="wide")


def run():
    st.title("JB Board")

    logdir = pathlib.Path(os.environ.get("logdir", DEFAULT_RESULT_DIR))
    session = Session(logdir)

    tab_names = ["Analysis", "Instance data", "Problem", "Solver"]
    tab_map = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

    with tab_map["Analysis"]:
        session.display_page("Analysis")

    with tab_map["Instance data"]:
        session.display_page("Instance data")

    with tab_map["Problem"]:
        session.display_page("Problem")

    with tab_map["Solver"]:
        session.display_page("Solver")


if __name__ == "__main__":
    run()
