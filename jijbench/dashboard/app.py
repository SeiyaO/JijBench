from __future__ import annotations

import streamlit as st

from jijbench.dashboard.session import Session


st.set_page_config(layout="wide")


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
