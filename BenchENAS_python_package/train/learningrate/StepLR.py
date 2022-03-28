# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import torch


class StepLR(BaseLearningRate):
    """StepLR
    """

    def __init__(self, **kwargs):
        super(StepLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

