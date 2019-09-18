# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Index(Function):


    def forward(self, a, idxs):
        return a[idxs]
    
    def backward_0(self, grad):
        a, axis = self.inputs
        output_grad = np.zeros_like(a, dtype=self.inputs[0].dtype.type)
        output_grad[axis] = grad
        return output_grad
