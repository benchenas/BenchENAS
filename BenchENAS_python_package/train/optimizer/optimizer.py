# coding=utf-8


class BaseOptimizer():
    """Abstract class for Optimizer
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_optimizer(self, weight_params, params):
        raise NotImplementedError()
