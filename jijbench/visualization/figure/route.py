from __future__ import annotations

from numbers import Number

import numpy.typing as npt
import plotly
import plotly.express as px
import plotly.graph_objs as go

from jijbench.visualization.figure.interface_plotly import Figure


class Route(Figure):
    """Visualize route by plotly.

    Attributes:
        nodes (dict[str | Number, tuple[Number, Number]]): the dict of nodes. the key is label, and the value is tuple of two coordinates (x, y).
        routes (dict[str, list[str | Number]]): the dict of routes. the key is route name, and the value is the list of node label represents the order of visits.
        savefig (bool): If True, save the figure.
        savedir (str): the directory to save the figure.
        savescale (int): corresponds to the resolution of the figure.

    Example:
        ```python
        from jijbench.visualization import Route

        route = Route(savefig=True)
        route.add_nodes(
            node_pos={
                0: (0.0, 0.0),
                1: (1.0, 2.0),
                2: (3.0, 4.0),
                3: (-1.0, -2.0),
                4: (-1.0, 0.0),
            }
        )
        route.add_route(route=[0, 1, 2, 0])
        route.add_route(route=[0, 3, 4, 0])
        fig = route.create_figure(
            title="Title",
            height=1200,
            width=1200,
            xaxis_title="xaxis",
            yaxis_title="yaxis",
            shownodelabel=True,
            showlegend=True,
            savedir=".",
            savename="Savename",
            savescale=1,
        )
        fig.show()
        ```

        You can change the appearance of the figure by `update_layout` method.
        The example below changes the fontsize of the title.
        For other settings, please refer to the plotly reference.

        ```python
        fig.update_layout(title_font={"size": 30})
        fig.show()
            ```

    """

    def __init__(
        self,
        savefig: bool = True,
        savedir: str = ".",
        savescale: int = 2,
    ) -> None:
        """Initialize the Route instance with the given parameters.

        Args:
            savefig (bool): If True, save the figure. Defaults to True.
            savedir (str): the directory to save the figure. Defaults to `.`.
            savescale (int): corresponds to the resolution of the figure. Defaults to 2.
        """
        self._savefig = savefig
        self._savedir = savedir
        self._savescale = savescale
        self._nodes = {}
        self._routes = {}

    def add_nodes(
        self,
        node_pos: dict[str | Number, tuple[Number, Number]],
    ) -> None:
        """Add nodes to nodes attribute.

        Args:
            node_pos (dict[str | Number, tuple[Number, Number]]): the dict of nodes. the key is label, and the value is tuple of two coordinates (x, y).
        """

        self._nodes.update(node_pos)

    def add_route(
        self,
        route: list[str | Number],
        route_name: str | None = None,
    ) -> None:
        """Add route to routes attribute.

        Args:
            route (list[str | Number]): the list of node label represents the order of visits.
        """
        for node in route:
            if node not in self._nodes:
                raise ValueError(
                    f"node {node} is not in nodes. Please add node by add_nodes method."
                )
        if route_name is None:
            route_name = f"route{len(self._routes)}"
        self._routes[route_name] = route

    def create_figure(
        self,
        title: str | None = None,
        height: Number | None = None,
        width: Number | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        shownodelabel: bool | None = None,
        showlegend: bool | None = None,
        savefig: bool | None = None,
        savedir: str | None = None,
        savename: str | None = None,
        savescale: int | None = None,
    ) -> plotly.graph_objects.Figure:
        """Create a figure using plotly.

        Args:
            title (str | None): the title of the figure. If Defaults None is given, it will be set to Route.
            height (Number | None): the height of the figure. If Defaults None is given, it will be set to 600.
            width (Number | None): the width of the figure. If Defaults None is given, it will be set to 600.
            xaxis_title (str | None): the title of the x axis. If Defaults None is given, it will be set to x.
            yaxis_title (str | None): the title of the y axis. If Defaults None is given, it will be set to y.
            shownodelabel (bool | None): If True, show the node label. If the default None is given, it will be set to False.
            showlegend (bool | None): If True, show the legend. If the default None is given, it will be set to True.
            savefig (bool | None): If True, save the figure. If the default None is given, the value given to the constructor will be used.
            savedir (str | None): the directory to save the figure. If the default None is given, the value given to the constructor will be used.
            savename (str | None): the name of the figure. Since it is automatically saved as png, no extension is required.
                If Defaults None is given, it will be same as `title`.
            savescale (int | None): corresponds to the resolution of the figure. If the default None is given, the value given to the constructor will be used.
        """
        if title is None:
            title = "Route"
        if height is None:
            height = 600
        if width is None:
            width = 600
        if xaxis_title is None:
            xaxis_title = "x"
        if yaxis_title is None:
            yaxis_title = "y"
        if shownodelabel is None:
            shownodelabel = False
        if showlegend is None:
            showlegend = True
        if savefig is None:
            savefig = self._savefig
        if savedir is None:
            savedir = self._savedir
        if savename is None:
            savename = title
        if savescale is None:
            savescale = self._savescale

        fig = go.Figure()
        node_labels, x_node, y_node = self._get_node_coordinate()
        text = node_labels if shownodelabel else None
        hovertext = [
            f"label={label}<br>{xaxis_title}={x}<br>{yaxis_title}={y}"
            for label, x, y in zip(node_labels, x_node, y_node)
        ]
        # plot node
        fig.add_trace(
            go.Scatter(
                x=x_node,
                y=y_node,
                mode="markers+text",
                text=text,
                textposition="top center",
                hovertext=hovertext,
                hoverinfo="text",
                name="node",
            )
        )

        for route_name in self._routes.keys():
            x_route, y_route = self._get_routes_coordinate(route_name)
            # plot route
            fig.add_trace(
                go.Scatter(
                    x=x_route,
                    y=y_route,
                    mode="lines",
                    name=route_name,
                )
            )

        fig.update_layout(title=title)
        fig.update_layout(
            height=height,
            width=width,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=showlegend,
            template="plotly_white",  # Change background color from default color to white
            title={  # Change the position of the title from the default top left to center top
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family="Arial"),  # Change font to Arial
            xaxis=dict(
                linewidth=1, mirror=True, linecolor="black"
            ),  # Draw a line around the chart area
            yaxis=dict(
                ticks="outside", linewidth=1, mirror=True, linecolor="black"
            ),  # Draw a line around the chart area
        )

        if savefig:
            fig.write_image(f"{savedir}/{savename}.png", scale=savescale)

        return fig

    def _get_node_coordinate(
        self,
    ) -> tuple[list[str | Number], list[Number], list[Number]]:
        """Formats data in nodes attributes for plotly.

        Example:
            Let us consider the following nodes:
                self._nodes = {0: (0.1, 0.2), 1: (0.3, 0.4)}
            Then, returns the following:
                labels = [0, 1]
                x = [0.1, 0.3]
                y = [0.2, 0.4]
        """
        labels, x, y = [], [], []
        for label, pos in self._nodes.items():
            labels.append(label)
            x.append(pos[0])
            y.append(pos[1])
        return labels, x, y

    def _get_routes_coordinate(
        self, route_name: str
    ) -> tuple[list[Number], list[Number]]:
        """Formats data in routes attributes for plotly.

        Example:
            Let us consider the following nodes and routes:
                self._nodes = {0: (0.0, 0.0), 1: (1.0, 2.0), 2: (3.0, 4.0)}
                self._routes = {"route_name": [0, 1, 2, 0]}
            Then, returns the following for route_name="route_name":
                x = [0.0, 1.0, 3.0, 0.0]
                y = [0.0, 2.0, 4.0, 0.0]
        """
        nodes = self._nodes
        route = self._routes[route_name]
        x, y = [], []
        for node_id in route:
            x.append(nodes[node_id][0])
            y.append(nodes[node_id][1])
        return x, y

    @property
    def nodes(self):
        return self._nodes

    @property
    def routes(self):
        return self._routes

    @property
    def savefig(self):
        return self._savefig

    @property
    def savedir(self):
        return self._savedir

    @property
    def savescale(self):
        return self._savescale
