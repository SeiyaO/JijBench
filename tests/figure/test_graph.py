import matplotlib
import networkx as nx
import numpy as np
import pytest
from matplotlib import axes, collections, figure

from jijbench.visualization import Graph, GraphType

params = {
    "undirected case": ([[0, 1], [1, 2]], GraphType.UNDIRECTED, nx.Graph),
    "directed case": ([[0, 1], [1, 2]], GraphType.DIRECTED, nx.DiGraph),
}


@pytest.mark.parametrize(
    "edge_list, graphtype, expect_type",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_from_edge_list(edge_list, graphtype, expect_type):
    graph = Graph.from_edge_list(edge_list, graphtype)
    G = graph.G

    assert type(G) == expect_type
    assert len(G.edges()) == 2


params = {
    "undirected case": ([[-1, 1], [1, -1]], GraphType.UNDIRECTED, nx.Graph, 1),
    "directed case": ([[-1, 1], [2, -1]], GraphType.DIRECTED, nx.DiGraph, 2),
    "numpy case": (np.array([[-1, 1], [1, -1]]), GraphType.UNDIRECTED, nx.Graph, 1),
}


@pytest.mark.parametrize(
    "distance_matrix, graphtype, expect_type, expect_edge_num",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_from_distance_matrix(
    distance_matrix, graphtype, expect_type, expect_edge_num
):
    graph = Graph.from_distance_matrix(distance_matrix, graphtype)
    G = graph.G

    assert type(G) == expect_type
    assert len(G.edges()) == expect_edge_num


def test_graph_fig_ax_attribute():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    assert type(fig) == figure.Figure
    assert type(ax) == axes.Subplot


def test_graph_fig_ax_attribute_before_show():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)

    with pytest.raises(AttributeError):
        graph.fig_ax


params = {
    "give argument": ("title", "title"),
    "default": (None, "graph"),
}


@pytest.mark.parametrize(
    "title, expect",
    list(params.values()),
    ids=list(params.keys()),
)
def test_graph_show_arg_title(title, expect):
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(title=title)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    success_show_title = False
    for obj in fig.texts:
        actual_title = obj.get_text()
        if actual_title == expect:
            success_show_title = True

    assert success_show_title


def test_graph_show_arg_figsize():
    figwidth, figheight = 8, 4

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(figsize=tuple([figwidth, figheight]))
    fig, ax = graph.fig_ax

    assert fig.get_figwidth() == 8
    assert fig.get_figheight() == 4


def test_graph_show_node():
    graph = Graph.from_edge_list([[1, 2], [2, 3]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    # Check that the children of ax contain the expected node information
    expect_node_num = 3
    success_show_node = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_node_num = obj.get_offsets().data.shape[0]
            if (np.abs(actual_node_num - expect_node_num) < 0.0001).all():
                success_show_node = True
    assert success_show_node


def test_graph_show_arg_node_pos():
    pos1 = np.array([1, 1])
    pos2 = np.array([-1, -1])
    node_pos = {1: pos1, 2: pos2}

    # Check that the children of ax contain the expected node position information
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    expect_pos = np.vstack([pos1, pos2])
    success_set_pos = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_pos = obj.get_offsets().data
            if (np.abs(actual_pos - expect_pos) < 0.0001).all():
                success_set_pos = True
    assert success_set_pos


def test_graph_show_arg_node_pos_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    # Check that the children of ax contain the expected node position information
    default_pos = nx.spring_layout(graph.G, seed=1)
    expect_pos = np.vstack(list(default_pos.values()))
    success_set_pos = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_pos = obj.get_offsets().data
            if (np.abs(actual_pos - expect_pos) < 0.0001).all():
                success_set_pos = True
    assert success_set_pos


def test_graph_show_arg_node_color():
    node_color = ["r", "b"]

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_color=node_color)
    fig, ax = graph.fig_ax

    # Check that the children of ax contain the expected node color information
    expect_color_node1 = np.array(matplotlib.colors.to_rgb("r"))
    success_coloring_node1 = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color_node1 = obj.get_facecolor()[0][:-1]
            if (np.abs(actual_color_node1 - expect_color_node1) < 0.0001).all():
                success_coloring_node1 = True
    assert success_coloring_node1

    expect_color_node2 = np.array(matplotlib.colors.to_rgb("b"))
    success_coloring_node2 = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color_node2 = obj.get_facecolor()[1][:-1]
            if (np.abs(actual_color_node2 - expect_color_node2) < 0.0001).all():
                success_coloring_node2 = True
    assert success_coloring_node2


def test_graph_show_arg_node_color_default():
    default_color = "#1f78b4"

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_color = np.array(matplotlib.colors.to_rgb(default_color))

    # Check that the children of ax contain the expected node color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.PathCollection:
            actual_color = obj.get_facecolor()[0][:-1]
            if (np.abs(actual_color - expect_color) < 0.0001).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_node_labels():
    node1, node2 = 1, 2
    node_labels = {node1: "node1", node2: "node2"}
    node_pos = {node1: np.array([1, 1]), node2: np.array([-1, -1])}

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos, node_labels=node_labels)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    expect_info_node1 = (node_pos[node1][0], node_pos[node1][1], "node1")
    success_show_node1_label = False
    for obj in ax.texts:
        actual_info_node1 = (*obj.get_position(), obj.get_text())
        if actual_info_node1 == expect_info_node1:
            success_show_node1_label = True
    assert success_show_node1_label

    expect_info_node2 = (node_pos[node2][0], node_pos[node2][1], "node2")
    success_show_node2_label = False
    for obj in ax.texts:
        actual_info_node2 = (*obj.get_position(), obj.get_text())
        if actual_info_node2 == expect_info_node2:
            success_show_node2_label = True
    assert success_show_node2_label


def test_graph_show_arg_node_labels_default():
    node1, node2 = 1, 2
    node_pos = {node1: np.array([1, 1]), node2: np.array([-1, -1])}

    graph = Graph.from_edge_list([[node1, node2]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected node label information
    expect_info_node1 = (node_pos[node1][0], node_pos[node1][1], str(node1))
    success_show_node1_label = False
    for obj in ax.texts:
        actual_info_node1 = (*obj.get_position(), obj.get_text())
        if actual_info_node1 == expect_info_node1:
            success_show_node1_label = True
    assert success_show_node1_label

    expect_info_node2 = (node_pos[node2][0], node_pos[node2][1], str(node2))
    success_show_node2_label = False
    for obj in ax.texts:
        actual_info_node2 = (*obj.get_position(), obj.get_text())
        if actual_info_node2 == expect_info_node2:
            success_show_node2_label = True
    assert success_show_node2_label


def test_graph_show_arg_edge():
    edge_list = [[1, 2], [2, 3]]

    graph = Graph.from_edge_list(edge_list, GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_edge_num = 2

    # Check that the children of ax contain the expected edge information
    success_show_edge = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_edge_num = len(obj.get_paths())
            if actual_edge_num == expect_edge_num:
                success_show_edge = True
    assert success_show_edge


def test_graph_show_arg_edge_color():
    edge_color = ["r"]

    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show(edge_color=edge_color)
    fig, ax = graph.fig_ax

    expect_color = np.array([1.0, 0.0, 0.0])

    # Check that the children of ax contain the expected edge color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_color = obj.get_color()[0][:-1]
            if (actual_color == expect_color).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_edge_color_default():
    graph = Graph.from_edge_list([[1, 2]], GraphType.UNDIRECTED)
    graph.show()
    fig, ax = graph.fig_ax

    expect_color = np.array([0.0, 0.0, 0.0])  # "black"

    # Check that the children of ax contain the expected edge color information
    success_coloring = False
    for obj in ax.get_children():
        if type(obj) == collections.LineCollection:
            actual_color = obj.get_color()[0][:-1]
            if (actual_color == expect_color).all():
                success_coloring = True
    assert success_coloring


def test_graph_show_arg_edge_labels():
    node_pos = {
        0: np.array([-1, -1]),
        1: np.array([1, 1]),
    }
    edge_labels = {(0, 1): "edge"}

    graph = Graph.from_edge_list([[0, 1]], GraphType.UNDIRECTED)
    graph.show(node_pos=node_pos, edge_labels=edge_labels)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected edge label information
    expect_label_x, expect_label_y = (node_pos[0] + node_pos[1]) / 2
    expect_label = edge_labels[(0, 1)]
    expect_info = (expect_label_x, expect_label_y, expect_label)

    success_show_edge_label = False
    for obj in ax.texts:
        actual_info = (*obj.get_position(), obj.get_text())
        if actual_info == expect_info:
            success_show_edge_label = True

    assert success_show_edge_label


def test_graph_show_arg_edge_labels_default_weighted_edge_case():
    node_pos = {0: np.array([1, 1]), 1: np.array([-1, -1])}
    weight = 5

    graph = Graph.from_distance_matrix(
        [[-1, weight], [weight, -1]], GraphType.UNDIRECTED
    )
    graph.show(node_pos=node_pos)
    fig, ax = graph.fig_ax

    # Check that the ax.texts contain the expected edge label information
    expect_label_x, expect_label_y = (node_pos[0] + node_pos[1]) / 2
    expect_label = str(weight)
    expect_info = (expect_label_x, expect_label_y, expect_label)

    success_show_edge_label = False
    for obj in ax.texts:
        actual_info = (*obj.get_position(), obj.get_text())
        if actual_info == expect_info:
            success_show_edge_label = True

    assert success_show_edge_label
