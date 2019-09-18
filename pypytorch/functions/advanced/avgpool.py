# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce
from pypytorch.functions.function import Function
from pypytorch import utils



class AvgPool2d(Function):

    def __init__(self, stride, padding):
        super(AvgPool2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.max_coords = []

    def forward(self, x, kernel_size):
        data = x
        data = utils.make_padding(data, self.padding)
        batch, channels, height, width = data.shape
        output_height = int((height - kernel_size[0]) / self.stride[0] + 1)
        output_width = int((width - kernel_size[1]) / self.stride[1] + 1)

        output = np.zeros((batch, channels, output_height, output_width))
        
        for b in range(batch):
            for c in range(channels):
                for y in range(output_height):
                    y_min = y * self.stride[0]
                    y_max = y_min + kernel_size[0]
                    for x in range(output_width):
                        x_min = x * self.stride[1]
                        x_max = x_min + kernel_size[1]
                        field = data[b, c, y_min:y_max, x_min:x_max]

                        output[b, c, y, x] = np.mean(field)

        return output

    def backward_0(self, grad):
        """
        Parameters
        ----------
        grad : ndarray
        """
        x, kernel_size = self.inputs
        data = x
        batch, channels, height, width = data.shape
        output_grad = np.zeros_like(data)
        output_grad = utils.make_padding(output_grad, self.padding)
        output_height = int((height - kernel_size[0]) / self.stride[0] + 1)
        output_width = int((width - kernel_size[1]) / self.stride[1] + 1)
        area = reduce(lambda x, y: x * y, kernel_size)
        for b in range(batch):
            for c in range(channels):
                for y in range(output_height):
                    y_min = y * self.stride[0]
                    y_max = y_min + kernel_size[0]
                    for x in range(output_width):
                        x_min = x * self.stride[1]
                        x_max = x_min + kernel_size[1]
                        output_grad[b, c, y_min:y_max, x_min:x_max] = grad[b, c, y, x] / area
        return output_grad
