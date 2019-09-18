# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class Log(Function):


    def forward(self, a):
        """
        Parameters
        ----------
        a : tensor
        """
        return np.log(a)
    
    def backward_0(self, grad):
        a, = self.inputs
        return grad / a
    

    
