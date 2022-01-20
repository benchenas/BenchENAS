# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import torch


class RMSprop(BaseOptimizer):
    """RMSprop optimizer
    """

    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return torch.optim.RMSprop(weight_params, lr=self.lr)
