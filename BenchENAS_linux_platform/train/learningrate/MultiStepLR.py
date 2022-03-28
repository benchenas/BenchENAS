# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import torch


class MultiStepLR(BaseLearningRate):
    """MultiStepLR
    """

    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,15,25,30], gamma=0.1)

