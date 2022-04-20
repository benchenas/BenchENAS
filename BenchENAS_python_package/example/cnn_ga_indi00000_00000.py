"""
2022-04-11  20:46:06
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from datetime import datetime
import multiprocessing

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()

        #conv unit
        self.conv_3_128 = BasicBlock(in_planes=3, planes=128)
        self.conv_128_64 = BasicBlock(in_planes=128, planes=64)
        self.conv_64_128 = BasicBlock(in_planes=64, planes=128)
        self.conv_128_256 = BasicBlock(in_planes=128, planes=256)
        self.conv_256_256 = BasicBlock(in_planes=256, planes=256)
        self.conv_256_128 = BasicBlock(in_planes=256, planes=128)
        self.conv_256_64 = BasicBlock(in_planes=256, planes=64)
        self.conv_64_64 = BasicBlock(in_planes=64, planes=64)
        self.conv_64_256 = BasicBlock(in_planes=64, planes=256)

        #linear unit
        self.linear = nn.Linear(2048,10)


    def forward(self, x):
        out_0 = self.conv_3_128(x)
        out_1 = F.max_pool2d(out_0, 2)
        out_2 = self.conv_128_64(out_1)
        out_3 = self.conv_64_128(out_2)
        out_4 = F.max_pool2d(out_3, 2)
        out_5 = self.conv_128_256(out_4)
        out_6 = self.conv_256_256(out_5)
        out_7 = self.conv_256_128(out_6)
        out_8 = self.conv_128_256(out_7)
        out_9 = self.conv_256_64(out_8)
        out_10 = self.conv_64_64(out_9)
        out_11 = F.max_pool2d(out_10, 2)
        out_12 = self.conv_64_256(out_11)
        out_13 = self.conv_256_256(out_12)
        out_14 = self.conv_256_128(out_13)
        out = out_14

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
