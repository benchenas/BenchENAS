"""
2022-04-18  19:44:42
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

c0 = 16

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

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        return x


class motif_1_1(nn.Module):
    def __init__(self, in_channel = c0):
        super(motif_1_1, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1_1 = nn.Conv2d(in_channel,in_channel,1,1)

    def forward(self, input):
        return self.relu(self.bn(self.conv1_1(input)))

class motif_1_2(nn.Module):
    def __init__(self, in_channels=c0):
        super(motif_1_2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.depwise_conv_3_3 = DepthwiseConv2d(in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, input):
        return self.relu(self.bn(self.depwise_conv_3_3(input)))

class motif_1_3(nn.Module):
    def __init__(self, in_channel=c0):
        super(motif_1_3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv3_3 = SeparableConv2d(in_channel,in_channel,3,1,1)

    def forward(self, input):
        return self.relu(self.bn(self.conv3_3(input)))


class motif_1_4(nn.Module):
    def __init__(self, in_channel=c0):
        super(motif_1_4, self).__init__()
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, input):
        return self.pool(input)


class motif_1_5(nn.Module):
    def __init__(self, in_channel=c0):
        super(motif_1_5, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, input):
        return self.pool(input)

class motif_1_6(nn.Module):
    def __init__(self, in_channel=c0):
        super(motif_1_6, self).__init__()

    def forward(self, input):
        return input

class motif_2_1(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_1, self).__init__()

    def forward(self, input):
        final = input
        return final


class motif_2_2(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_2, self).__init__()
        self.layer1 = motif_1_3(in_channel)
        self.layer2 = motif_1_6(in_channel)

    def forward(self, input):
        level1_1 = input
        level1_3 = input
        level1_2 = self.layer1(level1_1)
        final = level1_2
        level1_4 = self.layer2(level1_1)
        residual_level1_3 = level1_3
        level1_4 = level1_4 + residual_level1_3
        final = final + level1_4
        return final


class motif_2_3(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_3, self).__init__()
        self.layer1 = motif_1_6(in_channel)
        self.layer2 = motif_1_3(in_channel)
        self.layer3 = motif_1_4(in_channel)

    def forward(self, input):
        level1_1 = input
        level1_2 = self.layer1(level1_1)
        final = level1_2
        level1_3 = self.layer2(level1_1)
        level1_4 = self.layer3(level1_1)
        residual_level1_3 = level1_3
        level1_4 = level1_4 + residual_level1_3
        final = final + level1_4
        return final


class motif_2_4(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_4, self).__init__()
        self.layer1 = motif_1_3(in_channel)

    def forward(self, input):
        level1_1 = input
        level1_4 = self.layer1(level1_1)
        final = level1_4
        return final


class motif_2_5(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_5, self).__init__()
        self.layer1 = motif_1_3(in_channel)
        self.layer2 = motif_1_3(in_channel)
        self.layer3 = motif_1_2(in_channel)

    def forward(self, input):
        level1_1 = input
        level1_2 = self.layer1(level1_1)
        level1_3 = self.layer2(level1_2)
        level1_4 = self.layer3(level1_3)
        final = level1_4
        return final


class motif_2_6(nn.Module):
    def __init__(self, in_channel):
        super(motif_2_6, self).__init__()
        self.layer1 = motif_1_6(in_channel)
        self.layer2 = motif_1_1(in_channel)

    def forward(self, input):
        level1_1 = input
        level1_2 = self.layer1(level1_1)
        level1_4 = self.layer2(level1_2)
        final = level1_4
        return final


class motif_3_1(nn.Module):
    def __init__(self, in_channel):
        super(motif_3_1, self).__init__()
        self.layer1 = motif_2_2(in_channel)
        self.layer2 = motif_2_3(in_channel)
        self.layer3 = motif_2_1(in_channel)

    def forward(self, input):
        level2_1 = input
        level2_3 = self.layer1(level2_1)
        level2_4 = self.layer2(level2_3)
        final = level2_4
        level2_5 = self.layer3(level2_1)
        residual_level2_3 = level2_3
        level2_5 = level2_5 + residual_level2_3
        final = final + level2_5
        return final




class CellModel(nn.Module):
    def __init__(self, in_channel=c0):
        super(CellModel, self).__init__()
        self.in_channel = in_channel
        self.layer1 = motif_3_1(in_channel)


    def forward(self, input):
       return self.layer1(input)


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.conv3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.cellmodel1 = CellModel(16)
        self.cellmodel2 = CellModel(32)
        self.cellmodel3 = CellModel(64)
        self.sep_conv3x3_2_1 = SeparableConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.sep_conv3x3_2_2 = SeparableConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.sep_conv3x3_1 = SeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(4096, 10)

    def forward(self, input):
        x = self.conv3x3(input)
        x = self.cellmodel1(x)
        x = self.sep_conv3x3_2_1(x)
        x = self.cellmodel2(x)
        x = self.sep_conv3x3_2_2(x)
        x = self.cellmodel3(x)
        x = self.sep_conv3x3_1(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = nn.Softmax(1)(x)
        return x