from abc import ABCMeta, abstractmethod


class Figure(metaclass=ABCMeta):
    # Create a figure using plotly.
    @abstractmethod
    def create_figure(self):
        raise NotImplementedError
