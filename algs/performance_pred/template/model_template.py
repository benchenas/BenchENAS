"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os,argparse
import numpy as np

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #ANCHOR-generated_init


    def forward(self, x):
        #ANCHOR-generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out 

"""