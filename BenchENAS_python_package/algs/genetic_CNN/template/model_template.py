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

class ConvBlock(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1):
        super(ConvBlock,self).__init__()
        self.conv_1 = nn.Conv2d(ch_in,ch_out,kernel_size,stride=stride,padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)

    def  forward(self,x):
        out = F.relu(self.bn_1(self.conv_1(x)))
        return out



class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        #generated_init


    def forward(self, input):
        #generate_forward

        out = torch.flatten(input, 1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)
        return output
"""
