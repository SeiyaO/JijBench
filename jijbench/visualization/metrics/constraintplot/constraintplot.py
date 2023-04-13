from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes, figure
from matplotlib.ticker import MaxNLocator

import jijbench as jb
from jijbench.visualization.metrics.baseplot.baseplot import BasePlot


def _get_violations_dict(x: pd.Series) -> dict:
    """Get a dictionary of constraint violations from `pd.Series`.

    This function is intended to be used for `jb.Experiment.table` as example.

    Args:
        x (pd.Series): a Series of benchmark result. Expected to include information on constraint violations.

    Returns:
        dict: a dictionary of constraint violations

    Example:
        The code below get the dictionary of constraint violations for each row in experiment.

        ```python
        import jijbench as jb
        import jijzept as jz
        from jijbench.visualization.metrics.constraintplot.constraintplot import _get_violations_dict

        problem = jb.get_problem("TSP")
        instance_data = jb.get_instance_data("TSP")[0][1]
        multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
        multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

        config_path = "XX"
        sa_sampler = jz.JijSASampler(config=config_path)

        bench = jb.Benchmark(
            params = {
                "model": [problem],
                "feed_dict": [instance_data],
                "multipliers": [multipliers1, multipliers2],
            },
            solver = [sa_sampler.sample_model],
        )
        experiment = bench()
        metrics = experiment.table.apply(_get_violations_dict, axis=1)
        ```
    """
    return x.filter(regex="violations").to_dict()


class ConstraintPlot:
    def __init__(self, result: jb.Experiment) -> None:
        """Visualize the metrics of a benchmark result by boxplot.

        Attributes:
            result (jb.Experiment): a benchmark result.
        """
        self.result = result
        self.baseplot = BasePlot(result)

    def boxplot_violations(
        self,
        figsize: tuple[int | float] | None = None,
        title: str | list[str] | None = None,
        title_fontsize: float | None = None,
        constraint_name_fontsize: float | None = None,
        constraint_name_fontrotation: float | None = None,
        ylabel: str | None = None,
        ylabel_size: float | None = None,
        yticks: list[int | float] | None = None,
        **matplotlib_boxplot_kwargs,
    ) -> tuple[tuple[figure.Figure, axes.Subplot]]:
        """Draw a box and whisker plot of the constraint violations of `result` data using matplotlib.boxplot.

        The arguments are passed to matplotlib functions to change the appearance of the plot.
        matplotlib_boxplot_kwargs are passed to matplotlib.boxplot, and defaults to `{showmeans: True, whis: [0, 100]}`.
            showmeans=True shows mean values in markers.
            the outliers are not considered and whiskers match maxima and minima by whis=[0, 100].
        This method returns a figure and axes, so you can post-process them to change the appearance of the plot.
        See also the example below.

        Args:
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | list[str] | None): the title of figure. The default uses the indices of `result.table`.
            title_fontsize (float | None): the fontsize of the title.The default uses matplotlib's default value.
            constraint_name_fontsize (float | None): the fontsize of the constraint name (i.e. xticklabels). The default uses matplotlib's default value.
            constraint_name_fontrotation (float | None): the rotation angle of the constraint name in degree.The default uses matplotlib's default value.
            ylabel (str | None): the ylabel of figure. Defaults to "constraint violations".
            ylabel_size (float | None): the fontsize of the ylabel. The default uses matplotlib's default value.
            yticks (list[int | float] | None): the yticks of figure. Default to only integers by`MaxNLocator(integer=True)`.
            **matplotlib_boxplot_kwargs (dict): the parameter passed to matplotlib.boxplot. Defaults to `{showmeans: True, whis: [0, 100]}`.

        Returns:
            tuple[tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]]: A tuple of length equal to the number of rows in result. each element of is a tuple of figure and axes.


        Example:
            Below is the code to boxplot the constraint violations.
            In the first example, postprocessing the figure and axes changes the appearance of the plot.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            from jijbench.visualization import ConstraintPlot

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
            multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=15)
            sa_sampler = jz.JijSASampler(config=config_path)

            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": [multipliers1, multipliers2],
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            cplot = ConstraintPlot(result)
            fig_ax_tuple = cplot.boxplot_violations()

            # you can post-process figure and axes to change the appearance of the plot.
            for fig, ax in fig_ax_tuple:
                fig.suptitle("my title")
                display(fig)
            ```

            By using the `construct_experiment_from_samplesets function`,
            `boxplot_violations` can also be used for `jm.SampleSet` obtained without `JijBenchmark`.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijbench.visualization import ConstraintPlot
            from jijbench.visualization import construct_experiment_from_samplesets

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers = {"onehot_time": 0.003, "onehot_location": 0.003}

            config_path = "XX"
            sampler = jz.JijSASampler(config=config_path)
            sampleset = sampler.sample_model(model=problem, feed_dict=instance_data, multipliers=multipliers, search=False, num_reads=100)

            result = construct_experiment_from_samplesets(sampleset)
            cplot = ConstraintPlot(result)
            fig_ax_tuple = cplot.boxplot_violations()
            ```
        """
        if ylabel is None:
            ylabel = "constraint violations"
        if len(matplotlib_boxplot_kwargs) == 0:
            # Show the arithmetic means in boxplot.
            matplotlib_boxplot_kwargs["showmeans"] = True
            # Make boxplot whisker positions min and max.
            matplotlib_boxplot_kwargs["whis"] = [0, 100]
        fig_ax_tuple = self.baseplot.boxplot(
            f=_get_violations_dict,
            figsize=figsize,
            title=title,
            title_fontsize=title_fontsize,
            xticklabels_size=constraint_name_fontsize,
            xticklabels_rotation=constraint_name_fontrotation,
            ylabel=ylabel,
            ylabel_size=ylabel_size,
            yticks=yticks,
            **matplotlib_boxplot_kwargs,
        )

        # Add a horizontal line to indicate that the constraint is satisfied. (violation = 0)
        fig_ax_list = []
        for fig, ax in fig_ax_tuple:
            ax.axhline(0, xmin=0, xmax=1, color="gray", linestyle="dotted")
            fig_ax_list.append((fig, ax))
        return tuple(fig_ax_list)
