# -*- coding: utf-8 -*-

from pypytorch.functions.function import Function


class MM(Function):
    """

    Notes
    -----
    Matmul doesn't have broadcast problem
    """

    def forward(self, a, b):
        return a @ b
    
    def backward_0(self, grad):
        a, b = self.inputs
        return grad @ b.T
    
    def backward_1(self, grad):
        a, b = self.inputs
        return a.T @ grad
    