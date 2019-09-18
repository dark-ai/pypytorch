# -*- coding: utf-8 -*-

from pypytorch.functions.function import Function


class Add(Function):


    def forward(self, a, b):
        return a + b
    
    def backward_0(self, grad):
        return grad
    
    def backward_1(self, grad):
        return grad
