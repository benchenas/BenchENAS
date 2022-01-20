# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import torch


class SGD(BaseOptimizer):
    """SGD optimizer
    """

    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return torch.optim.SGD(weight_params, lr=self.lr)
