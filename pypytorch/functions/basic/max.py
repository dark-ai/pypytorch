# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Max(Function):

    def __init__(self):
        super(Max, self).__init__()
        self.idxs = []
        self.out = 0
    
    def forward(self, a, dim, keepdims):
        self.idxs = a.argmax(axis=dim)
        self.out = a.max(axis=dim, keepdims=keepdims)
        return self.out
    
    def backward_0(self, grad):
        
        a, dim, keepdims = self.inputs
        out_grad = np.zeros_like(a)
        # if dim is None:
        out_grad[a == self.out] = self.out
        return out_grad
