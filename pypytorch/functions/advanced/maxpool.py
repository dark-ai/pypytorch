# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function
from pypytorch import utils



class MaxPool2d(Function):

    def __init__(self, stride, padding):
        super(MaxPool2d, self).__init__()
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

                        y_max_values = np.max(field, axis=1)
                        y_idxs = np.argmax(field, axis=1)
                        x_idx = np.argmax(y_max_values, axis=0)

                        max_y = y_idxs[x_idx]
                        max_x = x_idx
                        max_value = field[max_y, max_x]

                        output[b, c, y, x] = max_value
                        self.max_coords.append([b, c, y_min + max_y, x_min + max_x, max_value])
        return output

    def backward_0(self, grad):
        """
        Parameters
        ----------
        grad : ndarray
        """
        x, kernel_size = self.inputs
        data = x
        output_grad = np.zeros_like(data)
        output_grad = utils.make_padding(output_grad, self.padding)
        grad = grad.reshape((-1, ))
        for i, coord in enumerate(self.max_coords):
            b, c, h, w, value = coord
            output_grad[b, c, h, w] = grad[i]
            # output_grad[b, c, h, w] = value
        return utils.unwrap_padding(output_grad, self.padding)