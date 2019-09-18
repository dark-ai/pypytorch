# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.functions.function import Function


class Pow(Function):


    def forward(self, a, b):
        return np.power(a, b)
    
    def backward_0(self, grad):
        a, b = self.inputs
        return grad * b * (np.power(a, b - 1))
    
    def backward_1(self, grad):
        """
        Parameters
        ----------
        grad : ndarray
        """
        a, b = self.inputs
        return grad * np.power(a, b) * np.log(a)
