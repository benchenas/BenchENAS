# coding=utf-8
from train.learningrate.learningrate import BaseLearningRate
import torch


class CosineAnnealingLR(BaseLearningRate):
    """CosineAnnealingLR
    """

    def __init__(self, **kwargs):
        super(CosineAnnealingLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.current_epoch))

