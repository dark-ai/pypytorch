# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function
from pypytorch import utils

class Sum(Function):


    def forward(self, *args):
        a, axis = utils.fetch_args(args, 2)
        return a.sum(axis=axis)
    
    def backward_0(self, grad):
        a, axis = self.inputs
        return grad * np.ones_like(a, dtype=self.raw_inputs[0].dtype.type)
