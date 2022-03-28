# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import torch

class Adam(BaseOptimizer):
    """Adam optimizer
    """
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return torch.optim.Adam(weight_params,lr = self.lr)
    
