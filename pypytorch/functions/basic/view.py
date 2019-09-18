# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function


class View(Function):


    def forward(self, a, size):
        return np.reshape(a, size)
    
    def backward_0(self, grad):
        a, size = self.inputs
        return np.reshape(grad, a.shape)
