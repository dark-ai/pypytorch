# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class ReLU(Function):


    def forward(self, a):
        return np.where(a < 0, 0, a)
    
    def backward_0(self, grad):
        a, = self.inputs
        grad = np.where(a < 0, 0, 1) * grad
        return grad


class Sigmoid(Function):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, a):
        self.output = 1 / (1 + np.exp(-a))
        return self.output
    
    def backward_0(self, grad):
        a = self.output
        return grad * a * (1 - a)


class Tanh(Function):


    def __init__(self):
        super(Tanh, self).__init__()
        self.output = None
    
    def forward(self, a):
        self.output = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
        return self.output

    def backward_0(self, grad):
        return grad * (1 - self.output ** 2)
