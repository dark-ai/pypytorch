# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function


class DropOut(Function):


    def __init__(self, prob=0.5, train=True):
        super(DropOut, self).__init__()
        assert 0 < prob <= 1, 'probability must be between (0, 1]'
        self.prob = prob
        self.train = train
        self.noise = None

    def forward(self, a):
        if self.train:
            self.noise = np.random.binomial(1, self.prob, size=a.shape)
            a = a * self.noise
        return a
    
    def backward_0(self, grad):
        return grad * self.noise


class DropOut2d(Function):


    def forward(self, a):
        pass
    
    def backward_0(self, grad):
        pass