# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function


class Transpose(Function):


    def forward(self, a, dims):
        return np.transpose(a, dims)
    
    def backward_0(self, grad):
        a, dims = self.inputs
        rev_dims = [None for i in dims]
        for i, dim in enumerate(dims):
            rev_dims[dim] = i
        return np.transpose(grad, rev_dims)
