# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import torch


class ExponentialLR(BaseLearningRate):
    """ExponentialLR
    """

    def __init__(self, **kwargs):
        super(ExponentialLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.2)

