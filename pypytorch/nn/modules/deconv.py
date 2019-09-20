# -*- coding: utf-8 -*-
import numpy as np

from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import utils
from pypytorch import functions
from pypytorch.nn.modules.conv import Conv2d



class DeConv2d(Conv2d):


    def __init__(self, in_ch, out_ch, kernel_size,
                padding=(0, 0), bias=True):
        super(Conv2d, self).__init__(in_ch, out_ch, kernel_size, \
                stride=(1, 1), padding=padding, dilation=(1, 1), bias=bias)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        self.padding = (up_down_padding, left_right_padding)
        return functions.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)