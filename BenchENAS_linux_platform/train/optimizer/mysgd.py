# coding=utf-8

from train.optimizer.optimizer import BaseOptimizer
import torch

class MySGD(BaseOptimizer):
    """SGD optimizer
    """
    def __init__(self, **kwargs):
        super(MySGD, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):

        # lr = 1e-2
        lr = 0.025
        # if current_epoch ==0: lr = 0.01
        # if current_epoch > 0: lr = 0.1
        # if current_epoch > 148: lr = 0.01
        # if current_epoch > 248: lr = 0.001



        return torch.optim.SGD(weight_params, lr=lr, momentum = 0.9, weight_decay=3e-4)
    
