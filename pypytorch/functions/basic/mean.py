# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Mean(Function):


    def forward(self, a, dim, keepdims):
        return a.mean(axis=dim, keepdims=keepdims)
    
    def backward_0(self, grad):
        a, dim, keepdims = self.inputs
        if dim is None:
            return grad * np.ones_like(a) / a.size
        num = a.shape[dim]
        return grad * np.ones_like(a) / num
