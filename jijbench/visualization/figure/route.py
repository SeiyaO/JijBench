from __future__ import annotations

from numbers import Number

import numpy.typing as npt
import plotly
import plotly.express as px
import plotly.graph_objs as go


class Route:
    def __init__(
        self,
        savefig: bool = True,
        savedir: str = ".",
        savescale: int = 2,
    ) -> None:
        self._savefig = savefig
        self._savedir = savedir
        self._savescale = savescale
        self._nodes = {}
        self._routes = {}

    def add_nodes(
        self,
        node_pos: dict[str | Number, tuple[Number, Number]],
    ) -> None:
        self._nodes.update(node_pos)

    def add_route(
        self,
        route: list[str | Number],
        route_name: str | None = None,
    ) -> None:
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
        savefig: bool | None = None,
        savedir: str | None = None,
        savename: str | None = None,
        savescale: int | None = None,
    ) -> plotly.graph_objects.Figure:
        if title is None:
            title = "Route"
        if height is None:
            height = 600
        if width is None:
            width = 600
        if savefig is None:
            savefig = self._savefig
        if savedir is None:
            savedir = self._savedir
        if savename is None:
            savename = title
        if savescale is None:
            savescale = self._savescale

        fig = go.Figure()
        x_node, y_node = self._get_node_coordinate()
        # plot node
        fig.add_trace(
            go.Scatter(
                x=x_node,
                y=y_node,
                mode="markers",
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

    def _get_node_coordinate(self) -> tuple[list[Number], list[Number]]:
        x, y = [], []
        for pos in self._nodes.values():
            x.append(pos[0])
            y.append(pos[1])
        return x, y

    def _get_routes_coordinate(
        self, route_name: str
    ) -> tuple[list[Number], list[Number]]:
        nodes = self._nodes
        route = self._routes[route_name]
        x, y = [], []
        for node_id in route:
            x.append(nodes[node_id][0])
            y.append(nodes[node_id][1])
        return x, y

    @property
    def savefig(self):
        return self._savefig

    @property
    def savedir(self):
        return self._savedir

    @property
    def savescale(self):
        return self._savescale

    @property
    def nodes(self):
        return self._nodes

    @property
    def routes(self):
        return self._routes
