# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function


class Concat(Function):


    def forward(self, a, b, dim):
        return np.concatenate((a, b), axis=dim)
    
    def backward_0(self, grad):
        a, b, dim = self.inputs
        return np.split(grad, (a.shape[dim], ), axis=dim)[0]

    def backward_1(self, grad):
        a, b, dim = self.inputs
        return np.split(grad, (a.shape[dim], ), axis=dim)[1]
