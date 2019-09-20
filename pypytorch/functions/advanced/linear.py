# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function
from pypytorch import utils


class Linear(Function):


    def forward(self, *args):
        a, b, c = utils.fetch_args(args, 3)
        if c is None:
            return a @ b
        return a @ b + c
    
    def backward_0(self, grad):
        a, b, c = utils.fetch_args(self.inputs, 3)
        return grad @ b.T

    def backward_1(self, grad):
        a, b, c = utils.fetch_args(self.inputs, 3)
        return a.T @ grad
    
    def backward_2(self, grad):
        a, b, c = self.inputs
        # print(a.shape, grad.shape)
        # return np.ones_like(a).T @ grad
        return grad
