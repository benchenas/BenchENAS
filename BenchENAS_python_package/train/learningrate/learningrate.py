# coding=utf-8


class BaseLearningRate():
    """Abstract class for Learning Rate
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_learning_rate(self):
        raise NotImplementedError()

