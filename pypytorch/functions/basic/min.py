# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Min(Function):

    def __init__(self):
        super(Min, self).__init__()
        self.idxs = []
        self.out = 0

    def forward(self, a, dim, keepdims):
        self.idxs = a.argmin(axis=dim)
        self.out = a.min(axis=dim, keepdims=keepdims)
        return self.out
    
    def backward_0(self, grad):
        a, dim, keepdims = self.inputs
        out_grad = np.zeros_like(a)
        out_grad[a == self.out] = self.out
        return out_grad
