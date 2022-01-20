"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import datetime

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class AsymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsymmetricConv2d, self).__init__()
        self.conv1x7 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=(1,7), stride=1,padding=(1,2))
        self.conv7x1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=(7,1),stride=1,padding=(2,1))

    def forward(self, x):
        x = self.conv1x7(x)
        x = self.conv7x1(x)
        return x

class normal_cell(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(normal_cell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        #normal_cell_init

    def forward(self, x, x_prev):
        input0 = self.conv_prev_1x1(x_prev)
        input1 = self.conv_1x1(x)

        #normal_cell_forward
        return final


class reduction_cell(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(reduction_cell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.conv_final = SeparableConv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        #reduction_cell_init

    def forward(self, input0, input1):
        input0 = self.conv_prev_1x1(input0)
        input1 = self.conv_1x1(input1)
        #reduction_cell_forward
        return self.conv_final(final)


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=0, stride=1,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.num_classes = 10
        self.last_linear = nn.Linear(2048, self.num_classes)
        self.normalcell = normal_cell(32, 32, 32, 32)
        self.reductioncell = reduction_cell(32, 32, 32, 32)
        self.num_normal_cells = 6
        #generated_init


    def features(self, input):
        prev_x, x = self.conv0(input), self.conv0(input)  #32 32 32

        for _ in range(self.num_normal_cells):
            new_x  = self.normalcell(x, prev_x)
            prev_x, x = x, new_x

        new_x  = self.reductioncell(x, prev_x)
        prev_x, x = new_x, new_x

        for _ in range(self.num_normal_cells):
            new_x  = self.normalcell(x, prev_x)
            prev_x, x = x, new_x

        new_x  = self.reductioncell(x, prev_x)
        prev_x, x = new_x, new_x

        for _ in range(self.num_normal_cells):
            new_x  = self.normalcell(x, prev_x)
            prev_x, x = x, new_x

        return x


    def logits(self, features):
        x = self.relu(features)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = self.last_linear(x)
        x = nn.Softmax(1)(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
"""
