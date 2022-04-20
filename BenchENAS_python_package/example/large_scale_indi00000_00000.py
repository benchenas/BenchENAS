import torch
import torch.nn as nn
import torch.functional as F


class EvoCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_2 = torch.nn.Linear(3, 10)

    def forward(self, input):
        x = {}
        x[0] = input
        x[0] = torch.nn.functional.interpolate(x[0], size = (32, 32))
        x[1] = x[0]
        x[1] = torch.squeeze(torch.squeeze(torch.nn.AdaptiveAvgPool2d((1, 1))(x[1]), 3), 2)
        x[2] = x[1]
        x[2] = self.unit_2(x[2])
        return x[2]