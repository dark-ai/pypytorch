# -*- coding: utf-8 -*-

from pypytorch.functions.function import Function


class Neg(Function):


    def forward(self, a):
        return -a
    
    def backward_0(self, grad):
        return -grad
