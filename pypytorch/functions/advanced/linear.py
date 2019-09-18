# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Linear(Function):


    def forward(self, a, b, c):
        return a @ b + c
    
    def backward_0(self, grad):
        a, b, c = self.inputs
        return grad @ b.T

    def backward_1(self, grad):
        a, b, c = self.inputs
        return a.T @ grad
    
    def backward_2(self, grad):
        a, b, c = self.inputs
        # print(a.shape, grad.shape)
        # return np.ones_like(a).T @ grad
        return grad
