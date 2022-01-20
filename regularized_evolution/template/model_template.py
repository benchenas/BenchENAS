"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Fit(nn.Module):
    def __init__(self, prev_filters, filters):
        super().__init__()
        self.relu = nn.ReLU()

        self.p1 = nn.Sequential(
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        #make sure there is no information loss
        self.p2 = nn.Sequential(
            nn.ConstantPad2d((0, 1, 0, 1), 0),
            nn.ConstantPad2d((-1, 0, -1, 0), 0),   #cropping
            nn.AvgPool2d(1, stride=2),
            nn.Conv2d(prev_filters, int(filters / 2), 1)
        )

        self.bn = nn.BatchNorm2d(filters)

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(prev_filters, filters, 1),
            nn.BatchNorm2d(filters)
        )

        self.filters = filters

    def forward(self, inputs):
        x, prev = inputs
        if prev is None:
            return x

        #image size does not match
        elif x.size(2) != prev.size(2):
            prev = self.relu(prev)
            p1 = self.p1(prev)
            p2 = self.p2(prev)
            prev = torch.cat([p1, p2], 1)
            prev = self.bn(prev)

        elif prev.size(1) != self.filters:
            prev = self.dim_reduce(prev)

        return prev


class NormalCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dem_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.fit = Fit(prev_in, output_channels)
        #normal_cell_init

    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dem_reduce(x)
        input0 = h
        input1 = prev
        #normal_cell_forward

        return final, x

class ReductionCell(nn.Module):

    def __init__(self, x_in, prev_in, output_channels):
        super().__init__()

        self.dim_reduce = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(x_in, output_channels, 1),
            nn.BatchNorm2d(output_channels)
        )
        self.fit = Fit(prev_in, output_channels)
        #reduction_cell_init


    def forward(self, x):
        x, prev = x
        prev = self.fit((x, prev))
        h = self.dim_reduce(x)
        input0 = h
        input1 = prev
        #reduction_cell_forward

        return self.conv_final(final), x


class Net(nn.Module):

    def __init__(self, repeat_cell_num, reduction_num, filters, stemfilter, class_num=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stemfilter, 3, padding=1, bias=False),
            nn.BatchNorm2d(stemfilter)
        )

        self.prev_filters = stemfilter
        self.x_filters = stemfilter
        self.filters = filters
        #concat
        self.cell_layers = self._make_layers(repeat_cell_num, reduction_num)

        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(self.filters*self.normal_concat, class_num)


    def _make_normal(self, block, repeat, output):
        layers = []
        for r in range(repeat):
            layers.append(block(self.x_filters, self.prev_filters, output))
            self.prev_filters = self.x_filters
            self.x_filters = output * self.normal_concat #concatenate 6 branches

        return layers

    def _make_reduction(self, block, output):
        reduction = block(self.x_filters, self.prev_filters, output)
        self.prev_filters = self.x_filters
        self.x_filters = output * self.reduction_concat #stack for 4 branches

        return reduction

    def _make_layers(self, repeat_cell_num, reduction_num):

        layers = []
        for i in range(reduction_num):

            layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))
            self.filters *= 2
            layers.append(self._make_reduction(ReductionCell, self.filters))

        layers.extend(self._make_normal(NormalCell, repeat_cell_num, self.filters))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.stem(x)
        prev = None
        x, prev = self.cell_layers((x, prev))
        x = self.relu(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generate_net

    def forward(self, x):
        return self.net(x)
"""
