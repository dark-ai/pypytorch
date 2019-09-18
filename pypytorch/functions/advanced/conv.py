# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function
from pypytorch import utils


def im2col(im, kernel_size, stride, padding):
    batch, channels, height, width = im.shape
    im = np.pad(im, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant')
    im = np.transpose(im, (1, 2, 3, 0))
    out_height = (height - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    out_width = (width - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    col = np.zeros((kernel_size[0] * kernel_size[1] * channels, out_height * out_width * batch), dtype='int64')

    col_idx = 0

    for y in range(out_height):
        y_img_start = y * stride[0]
        y_img_end = y_img_start + kernel_size[0]
        for x in range(out_width):
            x_img_start = x * stride[1]
            x_img_end = x_img_start + kernel_size[1]
            col[:, col_idx:col_idx + batch] = im[:, y_img_start:y_img_end, x_img_start:x_img_end, :].reshape((col.shape[0], -1))
            col_idx += batch
    return col


def _handle_padding(im, padding):
    if padding[0] == 0 and padding[1] == 0:
        return im
    if padding[0] == 0:
        return im[:, :, :, padding[1]:-padding[1]]
    if padding[1] == 0:
        return im[:, :, padding[0]:-padding[0], :]
    return im[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


def col2im(col, im, kernel_size, stride, padding):
    batch, channels, height, width = im.shape
    out_height = (height - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    out_width = (width - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    im = np.zeros_like(np.pad(im, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant'))
    im = np.transpose(im, (1, 2, 3, 0))

    col_idx = 0

    for y in range(out_height):
        y_img_start = y * stride[0]
        y_img_end = y_img_start + kernel_size[0]
        for x in range(out_width):
            x_img_start = x * stride[1]
            x_img_end = x_img_start + kernel_size[1]
            tmp = np.zeros_like(im)
            tmp[:, y_img_start:y_img_end, x_img_start:x_img_end, :] = col[:, col_idx:col_idx + batch].reshape(
                                                                        [-1, kernel_size[0], kernel_size[1], batch]
                                                                    )
            im += tmp
            col_idx += batch
    im = np.transpose(im, (3, 0, 1, 2))
    return _handle_padding(im, padding)


class Conv2d(Function):


    def __init__(self, stride=(1, 1), padding=(0, 0)):
        super(Conv2d, self).__init__()
        self.stride = utils.ensure_tuple_list(stride)
        assert self.stride[0] > 0 and self.stride[1] > 0,\
            'stride must be lt 0'
        self.padding = utils.ensure_tuple_list(padding)
        self._col = None

    def forward(self, x, weight, bias):
        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        batch, channels, height, width = x.shape
        assert len(bias.shape) == 2, 'bias dims is (filter_num, bias_value)'
        assert bias.shape[0] == filter_num, 'bias.dims()[0] must be eq filter_num'
        assert weight_channels == channels, 'weight_channels must eq channels'

        col = im2col(x, kernel_size, self.stride, self.padding)
        self._col = col
        out = weight.reshape(filter_num, -1) @ col + bias
        out_height = (height - kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_width = (width - kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        out = out.reshape(filter_num, out_height, out_width, batch)
        return out.transpose(3, 0, 1, 2)
    
    def backward_0(self, grad):
        x, weight, bias = self.inputs
        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        batch, channels, height, width = x.shape
        
        grad = grad.transpose(1, 2, 3, 0).reshape(filter_num, -1)
        weight = weight.reshape(filter_num, -1)
        grad = weight.T @ grad
        out = col2im(grad, x, kernel_size, self.stride, self.padding)
        return out
    
    def backward_1(self, grad):
        x, weight, bias = self.inputs
        filter_num, weight_channels, kernel_size = weight.shape[0], weight.shape[1], (weight.shape[2], weight.shape[3])
        batch, channels, height, width = x.shape

        grad = grad.transpose(1, 2, 3, 0).reshape(filter_num, -1)
        grad = grad @ self._col.T
        return grad.reshape(weight.shape)

    def backward_2(self, grad):
        x, weight, bias = self.inputs
        filter_num = weight.shape[0]
        grad = np.transpose(grad, (1, 2, 3, 0))
        grad = np.sum(grad, axis=(1, 2, 3))
        return grad.reshape(filter_num, -1)
