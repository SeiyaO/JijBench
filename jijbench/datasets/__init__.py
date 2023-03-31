from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)


from jijbench.datasets.problem import (
    TSP,
    TSPTW,
    BinPacking,  # get_problem,
    Knapsack,
    NurseScheduling,
)

# from jijbench.datasets.instance_data import get_instance_data
#
__all__ = [
    "BinPacking",
    "Knapsack",
    "TSP",
    "TSPTW",
    "NurseScheduling",
]
