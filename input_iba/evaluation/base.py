from abc import ABCMeta, abstractmethod


class BaseEvaluation(metaclass=ABCMeta):
    """
    Base class for all evaluation methods
    get attribution map and img as input, returns a dictionary contains
    evaluation result
    """

    @abstractmethod
    def evaluate(self, heatmap, *args, **kwargs) -> dict:
        raise NotImplementedError
