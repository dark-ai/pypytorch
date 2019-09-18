# -*- coding: utf-8 -*-

from pypytorch.functions.function import Function


class Mul(Function):


    def forward(self, a, b):
        return a * b

    def backward_0(self, grad):
        a, b = self.inputs
        return grad * b

    
    def backward_1(self, grad):
        a, b = self.inputs
        return grad * a
