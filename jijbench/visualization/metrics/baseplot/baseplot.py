from __future__ import annotations

from typing import Callable, cast

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import jijbench as jb
from jijbench.visualization.metrics.utils import _create_fig_title_list


class BasePlot:
    def __init__(self, result: jb.Experiment) -> None:
        """Visualize the metrics of a benchmark result by boxplot.

        Attributes:
            result (jb.Experiment): a benchmark result.
        """
        self.result = result

    def boxplot(
        self,
        f: Callable,
        figsize: tuple[int | float] | None = None,
        title: str | list[str] | None = None,
        title_fontsize: float | None = None,
        xticklabels_size: float | None = None,
        xticklabels_rotation: float | None = None,
        ylabel: str | None = None,
        ylabel_size: float | None = None,
        yticks: list[int | float] | None = None,
        **matplotlib_boxplot_kwargs,
    ):
        """Draw a box and whisker plot of the metrics based on `result` data using matplotlib.boxplot.

        This method applies the function f to the result (i.e. `jb.Experiment`) to get the metrics (pd.Series), and draw boxplot of this metrics.
        This metrics series calculated as metrics = self.result.table.apply(f, axis=1) in this method assumes the following structure.
            The length is equal to the number of rows in `result.table`.
            the element is a dictionary where the key is the name of each boxplot and the value is the np.array of the boxplot data.
        This method returns a figure and axes, so you can post-process them to change the appearance of the plot.
        See also the example below.

        Args:
            f (Callable): Callalbe to apply to table and get metrics.
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | list[str] | None): the title of figure. The default uses the indices of `result.table`.
            title_fontsize (float | None): the fontsize of the title.The default uses matplotlib's default value.
            xticklabels_size (float | None): the fontsize of the xticklabels (i.e. the name of each boxplot). The default uses matplotlib's default value.
            xticklabels_rotation (float | None): the rotation angle of the xticklabels in degree.The default uses matplotlib's default value.
            ylabel (str | None): the ylabel of figure. Defaults to None.
            ylabel_size (float | None): the fontsize of the ylabel. The default uses matplotlib's default value.
            yticks (list[int | float] | None): the yticks of figure. Default to only integers by`MaxNLocator(integer=True)`.
            **matplotlib_boxplot_kwargs (dict): the parameter passed to matplotlib.boxplot.

        Returns:
            tuple[tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]]: A tuple of length equal to the number of rows in result. each element of is a tuple of figure and axes.

        Example:
            The code below draws a boxplot of violations of each constraint.
            Note that result (i.e. `jb.Experiment`) holds violations for each constraint in np.array format.
            In the first example, postprocessing the figure and axes changes the appearance of the plot.

            ```python
            import jijbench as jb
            from jijbench.visualization import  BasePlot
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            import pandas as pd

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

            def get_violations_dict(x: pd.Series) -> dict:
                constraint_violations_indices = x.index[x.index.str.contains("violations")]
                return {index: x[index] for index in constraint_violations_indices}

            bplot = BasePlot(result)
            fig_ax_tuple = bplot.boxplot(f=get_violations_dict)

            # you can post-process figure and axes to change the appearance of the plot.
            for fig, ax in fig_ax_tuple:
                fig.suptitle("my title")
                display(fig)
            ```

            By using the `construct_experiment_from_samplesets function`,
            `boxplot` can also be used for `jm.SampleSet` obtained without `JijBenchmark`.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijbench.visualization import  BasePlot
            from jijbench.visualization import construct_experiment_from_samplesets
            import pandas as pd

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers = {"onehot_time": 0.003, "onehot_location": 0.003}

            config_path = "XX"
            sampler = jz.JijSASampler(config=config_path)
            sampleset = sampler.sample_model(model=problem, feed_dict=instance_data, multipliers=multipliers, num_reads=100)

            def get_violations_dict(x: pd.Series) -> dict:
                constraint_violations_indices = x.index[x.index.str.contains("violations")]
                return {index: x[index] for index in constraint_violations_indices}

            result = construct_experiment_from_samplesets(sampleset)
            bplot = BasePlot(result)
            fig_ax_tuple = bplot.boxplot(f=get_violations_dict)
            ```
        """
        metrics = self.result.table.apply(f, axis=1)
        title_list = _create_fig_title_list(metrics, title)

        fig_ax_list = []
        for i, (_, data) in enumerate(metrics.items()):
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title_list[i], fontsize=title_fontsize)
            ax.boxplot(data.values(), **matplotlib_boxplot_kwargs)
            ax.set_xticklabels(
                data.keys(), size=xticklabels_size, rotation=xticklabels_rotation
            )
            ylabel = cast("str", ylabel)
            ax.set_ylabel(ylabel, size=ylabel_size)
            if yticks is None:
                # make yticks integer only
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                ax.set_yticks(yticks)
            fig_ax_list.append((fig, ax))

        return tuple(fig_ax_list)
