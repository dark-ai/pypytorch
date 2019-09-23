# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function
from pypytorch import utils


def nearest_upsample(inputs, outputs):
    assert len(inputs.shape) == 4, 'shape must be (batch, channel, height, width) format'
    h_input, w_input = inputs.shape[2], inputs.shape[3]
    h_output, w_output = outputs.shape[2], outputs.shape[3]
    h_ratio = h_input / h_output
    w_ratio = w_input / w_output
    for i in range(h_output):
        for j in range(w_output):
            outputs[..., i, j] = inputs[..., int(h_ratio * i), int(w_ratio * j)]
    return outputs


class NearestUpsample(Function):


    def __init__(self, size):
        super(NearestUpsample, self).__init__()
        self.size = size
        self._x = None

    def forward(self, x):
        outputs = np.zeros((x.shape[0], x.shape[1]) + self.size)
        outputs = nearest_upsample(x, outputs)
        self._x = x
        return outputs

    def backward_0(self, grad):
        outputs = np.zeros((grad.shape[0], grad.shape[1], self._x.shape[2], self._x.shape[3]))
        grad = nearest_upsample(grad, outputs)
        return grad 
