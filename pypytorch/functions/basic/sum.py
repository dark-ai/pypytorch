# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Sum(Function):


    def forward(self, a, axis):
        return a.sum(axis=axis)
    
    def backward_0(self, grad):
        a, axis = self.inputs
        return grad * np.ones_like(a, dtype=self.raw_inputs[0].dtype.type)
