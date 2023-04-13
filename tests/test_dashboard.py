import typing as tp

import plotly.graph_objects as go
import pytest

from jijbench.consts.default import DEFAULT_PROBLEM_NAMES
from jijbench.dashboard.handlers.instance_data import (
    InstanceDataDirTree,
    _load_instance_data,
    _plot_box_for_instance_data,
    _plot_histogram_for_instance_data,
    _plot_violin_for_instance_data,
)
from jijbench.datasets.model import InstanceDataFileStorage


# InstanceDataDirTreeクラスのテスト
def test_instance_data_dir_tree_initialization():
    instance_data_dir_tree = InstanceDataDirTree()
    print(instance_data_dir_tree.nodes)

    assert set(instance_data_dir_tree.problem_names) == set(DEFAULT_PROBLEM_NAMES)

    for problem_name in DEFAULT_PROBLEM_NAMES:
        root = instance_data_dir_tree.node_map[problem_name]
        assert set(root.keys()) == {"label", "value", "children"}

        child = root["children"][0]
        assert set(child.keys()) == {"label", "value", "children"}
        assert child["label"] in ["small", "medium", "large"]
        assert child["value"] == f"{problem_name}&{child['label']}"

        leaf = child["children"][0]
        assert set(leaf.keys()) == {"label", "value"}
        assert leaf["label"].startswith("sample-")


@pytest.mark.parametrize("problem_name", DEFAULT_PROBLEM_NAMES)
def test_load_instance_data(problem_name: str):
    files = InstanceDataFileStorage(problem_name).get_files()
    data_map = _load_instance_data(files)

    for file in files:
        assert file.split("/")[-1] in data_map


@pytest.mark.parametrize(
    "plot_function",
    [
        _plot_histogram_for_instance_data,
        _plot_box_for_instance_data,
        _plot_violin_for_instance_data,
    ],
)
@pytest.mark.parametrize("problem_name", DEFAULT_PROBLEM_NAMES)
def test_plot_functions(plot_function: tp.Callable[..., tp.Any], problem_name: str):
    files = InstanceDataFileStorage(problem_name).get_files()
    data_map = _load_instance_data(files)

    for file in files:
        key = file.split("/")[-1]
        fig = plot_function(data_map[key])
        assert isinstance(fig, go.Figure)
