class BaseEvaluation(object):
    """
    Base class for all evaluation methods
    get attribution map and img as input, returns a dictionary contains evaluation result
    """
    def evaluate(self, heatmap, *args, **kwargs) -> dict:
        raise NotImplementedError