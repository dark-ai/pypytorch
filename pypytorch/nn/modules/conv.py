# -*- coding: utf-8 -*-

import math
import numpy as np

from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import utils
from pypytorch import functions


class Conv2d(Module):


    def __init__(self, in_ch, out_ch, kernel_size,
                stride=(1, 1), padding='VALID', dilation=(1, 1), bias=True):
        super(Conv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = utils.pair(kernel_size)
        self.stride = utils.pair(stride)
        assert self.stride[0] > 0 and self.stride[1] > 0,\
            'stride must be lt 0'
        if isinstance(padding, str):
            self.padding = padding
        else:
            self.padding = utils.pair(padding)
        
        assert self.stride[0] <= self.kernel_size[0] or self.stride[0] <= self.kernel_size[1],\
            'stride must be le kernel_size'
        self.weight = t.Tensor((out_ch, in_ch, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)
        self.bias = t.Tensor((out_ch, 1), requires_grad=True)
        
        if not bias:
            self.bias = None
        self.reset_parameters()
    
    def train(self):
        self.prepare_modules_for_train()
        self.weight.requires_grad = True
        if hasattr(self, 'bias') and getattr(self, 'bias') is not None:
            getattr(self, 'bias').requires_grad = True
    
    def eval(self):
        self.prepare_modules_for_train()
        self.weight.requires_grad = False
        if hasattr(self, 'bias') and getattr(self, 'bias') is not None:
            getattr(self, 'bias').requires_grad = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_ch * self.kernel_size[0] * self.kernel_size[1])
        self.weight.data = np.random.uniform(-stdv, stdv, self.weight.data.shape)
        if self.bias:
            self.bias.data = np.random.uniform(-stdv, stdv, self.bias.data.shape)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if isinstance(self.padding, str):
            if self.padding.upper() == 'SAME':
                up_down_padding = (height * (self.stride[0] - 1) - self.stride[0] + self.kernel_size[0]) // 2
                left_right_padding = (width * (self.stride[1] - 1) - self.stride[1] + self.kernel_size[1]) // 2
            else:
                up_down_padding = 0
                left_right_padding = 0
            self.padding = (up_down_padding, left_right_padding)
        return functions.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def __str__(self):
        return 'Conv2d(in_ch=%s, out_ch=%s, kernel_size=%s, stride=%s, padding=%s, bias=%s)'\
            % (self.in_ch, self.out_ch, self.kernel_size, self.stride, self.padding, True if self.bias is not None else False)

    def __repr__(self):
        return str(self)
