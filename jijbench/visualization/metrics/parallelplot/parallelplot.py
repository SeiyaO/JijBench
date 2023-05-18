from __future__ import annotations

import re
from numbers import Number
from typing import Callable, Literal

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

import jijbench as jb
from jijbench.exceptions.exceptions import UserFunctionFailedError
from jijbench.visualization.metrics.utils import (
    _df_has_number_array_column_target_name,
    _df_has_number_column_target_name,
    _df_has_valid_multipliers_column,
)

AXIS_LABEL_POS = Literal["top", "bottom"]


def _calc_mean_of_array(x: pd.Series, column_name: str) -> float:
    """
    Gets the mean of the elements in the specified column ,assuming this element is an np.ndarray.

    This function is intended to be used for `jb.Experiment.table`
    as experiment.table.apply(_calc_mean_of_array, axis=1, column_name=column_name)
    """
    num_occ = x["num_occurrences"]
    array = x[column_name]
    mean = np.sum(num_occ * array) / np.sum(num_occ)
    return mean


def _get_multiplier(x: pd.Series, constraint_name: str) -> float:
    """
    Gets the multiplier of given constraint name from `pd.Series`.

    This function is intended to be used for `jb.Experiment.table`
    as experiment.table.apply(_get_multiplier, axis=1, constraint_name=constraint_name)
    """
    multipliers = x["multipliers"]
    return multipliers[constraint_name]


class MetricsParallelPlot:
    def __init__(self, result: jb.Experiment) -> None:
        """Visualize the metrics of a benchmark result by parallelplot.

        Attributes:
            result (jb.Experiment): a benchmark result.
            parallelplot_axes_list (list[str]): a list of the names of all the metrics that can be plotted by `parallelplot_experiment`.
                None until `parallelplot_experiment` is called.
        """
        self.result = result

    def parallelplot_experiment(
        self,
        color_column_name: str | None = None,
        color_midpoint: float | None = None,
        additional_axes: list[str] | None = None,
        additional_axes_created_by_function: dict[str, Callable] | None = None,
        display_axes_list: list[str] | None = None,
        rename_map: dict[str, str] | None = None,
        axis_label_pos: AXIS_LABEL_POS | None = None,
        axis_label_fontsize: Number | None = None,
        title: str | None = None,
        height: Number | None = None,
        width: Number | None = None,
    ) -> plotly.graph_objects.Figure:
        """Plot the parallel plot of the experiment.

        This method creates a `plotly.graph_objects.Figure` instance and returns it.
        To view the figure, you need to call the show method of the returned instance.

        Args:
            color_column_name (str | None): the column name, and the values from this column are used to assign color to mark.
                Defaults to samplemean_total_violations or objective if those columns exist.
            color_midpoint (float | None): the midpoint of the color scale. Defaults to the mean of the color column value.
            additional_axes (list[str] | None): A list of column names for additional axes.
                The conditions for available column names are that they are elements of self.result.table.columns and that the values from the column is number.
                Defaults to None.
            additional_axes_created_by_function (dict[str, Callable]): A list of dict, where the key is the label of the axis and the value is the callable.
                The callable is applied to `self.result.table` as self.result.table.apply(callable, axis=1), and the result is added to axes.
                The callable takes a `pd.Series` and returns a number.
                Defaults to None.
            display_axes_list (list[str] | None): A list of labels for the displayed axes. This argument allows you to select and sort the axes to display.
                Check the `parallelplot_axes_list` attribute for available axes. Defaults to all axes.
            rename_map (dict[str, str] | None): A dictionary where the key is the original axis label and the value is the user-specified axis label.
                Check the original axis labels in the `parallelplot_axes_list` attribute.
                Defaults is None, the original axis labels is displayed.
            axis_label_pos (AXIS_LABEL_POS | None): the position of the axis label. Only "top" or "bottom" are accepted. Defaults to top.
            axis_label_fontsize (Number | None): the fontsize of the axis label. Defaults to None.
            title (str | None): the title of the plot. Defaults to None.
            height (Number | None): the height of the plot. Defaults to None.
            width (Number | None): the width of the plot. Defaults to None.

        Returns:
            plotly.graph_objects.Figure: the parallel plot of the experiment.

        Examples:
            The following example is the most basic usage. A parallel plot of the benchmark results is displayed.

            ```python
            from itertools import product
            import jijbench as jb
            from jijbench.visualization import MetricsParallelPlot
            import jijzept as jz

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]

            onehot_time_multipliers = [0.01, 0.1]
            onehot_location_multipliers = [0.01, 0.1]
            multipliers = [
                {"onehot_time": onehot_time_multiplier,
                "onehot_location": onehot_location_multiplier}
                for onehot_time_multiplier, onehot_location_multiplier in product(onehot_time_multipliers, onehot_location_multipliers)
            ]
            config_path = XX
            sa_sampler = jz.JijSASampler(config=config_path)
            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "multipliers": multipliers,
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            mp = MetricsParallelPlot(result)
            fig = mp.parallelplot_experiment()
            fig.show()
            ```

            You can change the appearance of the graph by performing the following operations on the `plotly.graph_objects.Figure` instance returned by `parallelplot_experiment`.
            The example below changes the fontsize of the range.
            For other operations, refer to the plotly reference, https://plotly.com/python/reference/parcoords/.

            ```python
            fig.update_traces(rangefont_size=15, selector=dict(type='parcoords'))
            fig.show()
            ```

            This example gives some arguments to `parallelplot_experiment`.
            `additional_axes` argument adds the execution_time column to the plot. This is the column of result.table and its element are number.
            `additional_axes_created_by_function` argument add the values calculated from result.table to the plot.
            `rename_map` insert line breaks for long data labels to make the charts easier to read. Line breaks are done with <br>.
            `axis_label_pos` is set to bottom to avoid the data label overlapping the figure due to line breaks.

            ```python
            from itertools import product
            import numpy as np
            import pandas as pd

            import jijbench as jb
            from jijbench.visualization import MetricsParallelPlot
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]

            onehot_time_multipliers = [0.01, 0.1]
            onehot_location_multipliers = [0.01, 0.1]
            multipliers = [
                {"onehot_time": onehot_time_multiplier,
                "onehot_location": onehot_location_multiplier}
                for onehot_time_multiplier, onehot_location_multiplier in product(onehot_time_multipliers, onehot_location_multipliers)
            ]
            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=30)
            sa_sampler = jz.JijSASampler(config=config_path)
            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": multipliers,
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()


            def get_num_reads_from_parameters(x: pd.Series) -> float:
                return x["parameters"].num_reads

            def calc_samplemean_energy(x: pd.Series) -> float:
                num_occ	= x["num_occurrences"]
                array = x["energy"]
                mean = np.sum(num_occ * array) / np.sum(num_occ)
                return mean

            mp = MetricsParallelPlot(result)
            fig = mp.parallelplot_experiment(
                additional_axes=["execution_time"],
                additional_axes_created_by_function={
                    "num_reads": get_num_reads_from_parameters,
                    "samplemean_energy": calc_samplemean_energy,
                },
                rename_map={
                    "onehot_time_multiplier": "onehot_time<br>multiplier",
                    "onehot_location_multiplier": "onehot_location<br>multiplier",
                    "samplemean_objective": "samplemean<br>objective",
                    "samplemean_onehot_time_violations": "samplemean<br>onehot_time<br>violations",
                    "samplemean_onehot_location_violations": "samplemean<br>onehot_location<br>violations",
                    "samplemean_total_violations": "samplemean<br>total_violations",
                },
                axis_label_pos="bottom",
            )

            ```

        """
        if additional_axes is None:
            additional_axes = []
        if additional_axes_created_by_function is None:
            additional_axes_created_by_function = {}
        if axis_label_pos is None:
            axis_label_pos = "top"
        if not (axis_label_pos in ["top", "bottom"]):
            raise ValueError(
                f"axis_label_pos must be 'top' or 'bottom', but {axis_label_pos} is given."
            )

        result_table = self.result.table

        # The key is a column name (str), and the value is the data of each column (pd.Series).
        data_to_create_df_parallelplot = {}

        # Displayed data specified by user
        for display_column in additional_axes:
            # Check if the column exists and the elements is number.
            if not _df_has_number_column_target_name(result_table, display_column):
                raise TypeError(
                    f"{display_column}is not a column with number elements."
                )
            data_to_create_df_parallelplot[display_column] = result_table[
                display_column
            ]

        # Data generated by user custom functions
        for column_name, func in additional_axes_created_by_function.items():
            # TODO: デコレータで書き直した方が可読性が上がると思われる。エラーレイズと返り値がnumberであることのチェックをデコレータ内で行う
            try:
                data_to_create_df_parallelplot[column_name] = result_table.apply(
                    func, axis=1
                )
            except Exception as e:
                msg = f'An error occurred inside your function. Please check "{func.__name__}" in additional_axes_created_by_function. -> {e}'
                raise UserFunctionFailedError(msg)
            for value in data_to_create_df_parallelplot[column_name].values:
                if not isinstance(value, Number):
                    raise TypeError(
                        f"{column_name} is not a column with number elements."
                    )

        # multiplires (If self.result has a valid multipliers column)
        if _df_has_valid_multipliers_column(result_table):
            data_to_create_df_parallelplot.update(
                result_table.filter(regex="multipliers").to_dict(orient="list")
            )

        # num_feasible
        if _df_has_number_column_target_name(result_table, "num_feasible"):
            data_to_create_df_parallelplot["num_feasible"] = result_table[
                "num_feasible"
            ]

        # objective
        if _df_has_number_array_column_target_name(result_table, "objective"):
            data_to_create_df_parallelplot["samplemean_objective"] = result_table.apply(
                _calc_mean_of_array, axis=1, column_name="objective"
            )

        # violations
        for violation_column_name in result_table.columns[
            result_table.columns.str.contains("violations")
        ]:
            if _df_has_number_array_column_target_name(
                result_table, violation_column_name
            ):
                data_to_create_df_parallelplot[
                    "samplemean_" + violation_column_name
                ] = result_table.apply(
                    _calc_mean_of_array,
                    axis=1,
                    column_name=violation_column_name,
                )

        self.df_parallelplot = df_parallelplot = pd.DataFrame(
            data_to_create_df_parallelplot
        )

        if display_axes_list is None:
            display_axes_list = self.parallelplot_axes_list
        self.df_parallelplot_displayed = df_parallelplot_displayed = df_parallelplot[
            display_axes_list
        ]

        if color_column_name is None:
            if "samplemean_total_violations" in df_parallelplot_displayed.columns:
                color_column_name = "samplemean_total_violations"
            elif "samplemean_objective" in df_parallelplot_displayed.columns:
                color_column_name = "samplemean_objective"

        if color_midpoint is None and color_column_name is not None:
            color_midpoint = df_parallelplot_displayed[color_column_name].mean()

        fig = px.parallel_coordinates(
            df_parallelplot_displayed.reset_index(drop=True),
            color=color_column_name,
            labels=rename_map,
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=color_midpoint,
            title=title,
            height=height,
            width=width,
        )
        fig.update_traces(labelside=axis_label_pos, selector=dict(type="parcoords"))
        fig.update_traces(
            labelfont_size=axis_label_fontsize, selector=dict(type="parcoords")
        )

        return fig

    @property
    def parallelplot_axes_list(self) -> list[str]:
        if hasattr(self, "df_parallelplot"):
            return list(self.df_parallelplot.columns)
        else:
            return []
