import numpy as np
import plotly
import pytest

from jijbench.visualization import Route


def test_add_nodes():
    route = Route()
    route.add_nodes(node_pos={0: (0.1, 0.2)})
    route.add_nodes(node_pos={1: (0.3, 0.4)})
    assert route.nodes[0] == (0.1, 0.2)
    assert route.nodes[1] == (0.3, 0.4)


def test_add_route():
    route = Route()
    route.add_nodes(
        node_pos={
            0: (0.0, 0.0),
            1: (1.0, 2.0),
            2: (3.0, 4.0),
            3: (5.0, 6.0),
            4: (7.0, 8.0),
            5: (9.0, 10.0),
            6: (11.0, 12.0),
        }
    )
    route.add_route(route=[0, 1, 2, 0])
    route.add_route(route=[0, 3, 4, 0])
    route.add_route(route=[0, 5, 6, 0], route_name="route_name")
    assert route.routes["route0"] == [0, 1, 2, 0]
    assert route.routes["route1"] == [0, 3, 4, 0]
    assert route.routes["route_name"] == [0, 5, 6, 0]


def test_add_route_with_invalid_node():
    route = Route()
    with pytest.raises(ValueError):
        route.add_route(route=[0, 1])


def test_get_node_coordinate():
    route = Route()
    route.add_nodes(node_pos={0: (0.1, 0.2), 1: (0.3, 0.4)})
    x, y = route._get_node_coordinate()
    assert x == [0.1, 0.3]
    assert y == [0.2, 0.4]


def test_get_routes_coordinate():
    route = Route()
    route.add_nodes(
        node_pos={
            0: (0.0, 0.0),
            1: (1.0, 2.0),
            2: (3.0, 4.0),
        }
    )
    route.add_route(route=[0, 1, 2, 0], route_name="route_name")
    x, y = route._get_routes_coordinate(route_name="route_name")
    assert x == [0.0, 1.0, 3.0, 0.0]
    assert y == [0.0, 2.0, 4.0, 0.0]


def test_create_figure_default_param():
    route = Route()
    route.add_nodes(
        node_pos={
            0: (0.0, 0.0),
            1: (1.0, 2.0),
            2: (3.0, 4.0),
            3: (-1.0, -2.0),
            4: (-3.0, -4.0),
        }
    )
    route.add_route(route=[0, 1, 2, 0])
    route.add_route(route=[0, 3, 4, 0])
    fig = route.create_figure()
    # fig.show()

    assert isinstance(fig, plotly.graph_objects.Figure)


# def test_use_all_param():
#     pass
