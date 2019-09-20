# -*- coding: utf-8 -*-

import math
import numpy as np

from pypytorch.nn.modules.module import Module
from pypytorch.nn.modules.conv import Conv2d
import pypytorch as t
from pypytorch import utils
from pypytorch import functions


class DeConv2d(Conv2d):


    def __init__(self, in_ch, out_ch, kernel_size,
                stride=(1, 1), padding=(0, 0), bias=True):
        super(DeConv2d, self).__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=(1, 1), bias=bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        return functions.deconv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def __str__(self):
        return 'DeConv2d(in_ch=%s, out_ch=%s, kernel_size=%s, stride=%s, padding=%s, bias=%s)'\
            % (self.in_ch, self.out_ch, self.kernel_size, self.stride, self.padding, True if self.bias is not None else False)

    def __repr__(self):
        return str(self)
