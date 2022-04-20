"""
2022-04-18  14:53:32
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

        #conv unit
        self.conv_input_s1_input = ConvBlock(3, 20, 3, 1)
        self.conv_s1_input_s1_1 = ConvBlock(20, 20, 3, 1)
        self.conv_s1_1_s1_3 = ConvBlock(20, 20, 3, 1)
        self.conv_final_s1_input = ConvBlock(20, 20, 3, 1)
        self.pool_0 = nn.MaxPool2d(2, stride=2)
        self.conv_input_s2_input = ConvBlock(20, 50, 3, 1)
        self.conv_s2_input_s2_1 = ConvBlock(50, 50, 3, 1)
        self.conv_s2_1_s2_2 = ConvBlock(50, 50, 3, 1)
        self.conv_s2_2_s2_3 = ConvBlock(50, 50, 3, 1)
        self.conv_s2_1_s2_4 = ConvBlock(50, 50, 3, 1)
        self.conv_final_s2_input = ConvBlock(50, 50, 3, 1)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv_input_s3_input = ConvBlock(50, 50, 3, 1)
        self.conv_s3_input_s3_1 = ConvBlock(50, 50, 3, 1)
        self.conv_s3_input_s3_3 = ConvBlock(50, 50, 3, 1)
        self.conv_s3_1_s3_2 = ConvBlock(50, 50, 3, 1)
        self.conv_s3_1_s3_4 = ConvBlock(50, 50, 3, 1)
        self.conv_s3_1_s3_5 = ConvBlock(50, 50, 3, 1)
        self.conv_final_s3_input = ConvBlock(50, 50, 3, 1)
        self.pool_2 = nn.MaxPool2d(2, stride=2)

        #linear unit
        self.linear = nn.Linear(800, 10)


    def forward(self, input):
        s1_input = self.conv_input_s1_input(input)
        s1_1 = self.conv_s1_input_s1_1(s1_input)
        s1_3 = self.conv_s1_1_s1_3(s1_1)
        final_s1 = s1_3
        input = self.conv_final_s1_input(final_s1)
        input = self.pool_0(input)
        s2_input = self.conv_input_s2_input(input)
        s2_1 = self.conv_s2_input_s2_1(s2_input)
        s2_2 = self.conv_s2_1_s2_2(s2_1)
        residual_s2_2 = s2_2
        s2_3 = self.conv_s2_2_s2_3(s2_2)
        residual_s2_3 = s2_3
        s2_4 = s2_1
        s2_4 = s2_4 + residual_s2_2
        s2_4 = s2_4 + residual_s2_3
        s2_4 = self.conv_s2_1_s2_4(s2_1)
        final_s2 = s2_4
        input = self.conv_final_s2_input(final_s2)
        input = self.pool_1(input)
        s3_input = self.conv_input_s3_input(input)
        s3_1 = self.conv_s3_input_s3_1(s3_input)
        s3_3 = self.conv_s3_input_s3_3(s3_input)
        residual_s3_3 = s3_3
        residual_s3_3 = s3_3
        s3_2 = self.conv_s3_1_s3_2(s3_1)
        residual_s3_2 = s3_2
        s3_4 = s3_1
        s3_4 = s3_4 + residual_s3_3
        s3_4 = self.conv_s3_1_s3_4(s3_1)
        final_s3 = s3_4
        s3_5 = s3_1
        s3_5 = s3_5 + residual_s3_2
        s3_5 = s3_5 + residual_s3_3
        s3_5 = self.conv_s3_1_s3_5(s3_1)
        final_s3 = final_s3 + s3_5
        input = self.conv_final_s3_input(final_s3)
        input = self.pool_2(input)

        out = torch.flatten(input, 1)
        out = self.linear(out)
        output = F.log_softmax(out, dim=1)
        return output