# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function
from pypytorch.functions.advanced import Conv2d
from pypytorch import utils


class DeConv2d(Conv2d):


    def __init__(self, stride=(1, 1), padding=(0, 0)):
        super(DeConv2d, self).__init__(stride=stride, padding=padding)
        self._col = None
        self._x = None

    def forward(self, *args):
        x, weight, bias = utils.fetch_args(args, 3)
        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        x = np.insert(np.insert(x, range(1, x.shape[-1]), [0] * self.stride[0], axis=3), \
                range(1, x.shape[-2]), [0] * self.stride[1], axis=2)
        self._x = x
        batch, channels, height, width = x.shape
        if bias is not None:
            assert len(bias.shape) == 2, 'bias dims is (filter_num, bias_value)'
            assert bias.shape[0] == filter_num, 'bias.dims()[0] must be eq filter_num'
        assert weight_channels == channels, 'weight_channels must eq channels'

        col = utils.im2col(x, kernel_size, self.stride, self.padding)
        self._col = col
        if bias is not None:
            out = weight.reshape(filter_num, -1) @ col + bias
        else:
            out = weight.reshape(filter_num, -1) @ col
        out_height = (height - kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (width - kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        out = out.reshape(filter_num, out_height, out_width, batch)
        return out.transpose(3, 0, 1, 2)
    
    def backward_0(self, grad):
        x, weight, bias = utils.fetch_args(self.inputs, 3)
        # x = self._x

        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        batch, channels, height, width = x.shape
        grad = grad.transpose(1, 2, 3, 0).reshape(filter_num, -1)
        weight = weight.reshape(filter_num, -1)
        grad = weight.T @ grad
        out = utils.col2im(grad, x, kernel_size, self.stride, self.padding)
        return out
    
    def backward_1(self, grad):
        x, weight, bias = utils.fetch_args(self.inputs, 3)
        # x = self._x
        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        batch, channels, height, width = x.shape

        grad = grad.transpose(1, 2, 3, 0).reshape(filter_num, -1)
        grad = grad @ self._col.T
        return grad.reshape(weight.shape)

    def backward_2(self, grad):
        x, weight, bias = self.inputs
        # x = self._x
        filter_num = weight.shape[0]
        grad = np.transpose(grad, (1, 2, 3, 0))
        grad = np.sum(grad, axis=(1, 2, 3))
        return grad.reshape(filter_num, -1)
