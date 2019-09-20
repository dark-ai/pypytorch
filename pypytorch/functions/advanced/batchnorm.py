# -*- coding: utf-8 -*-


import numpy as np
import pypytorch as ppt
from pypytorch.functions.function import Function


class BatchNorm2d(Function):


    def __init__(self, momentum=0.1, epsilon=1e-5, train=True):
        super(BatchNorm2d, self).__init__()
        self.mean = 0.0
        self.var = 0.0
        self.running_mean = 1.0
        self.running_var = 0.0
        self.momentum = momentum
        self.epsilon = epsilon
        self.x_hat = 0.0
        self.train = train

    def forward(self, x, weight, bias):
        if self.train:
            self.mean = x.mean(axis=0)
            self.var = x.var(axis=0)
            self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.var + (1 - self.momentum) * self.var
            self.x_hat = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.var + self.epsilon)
        y = self.x_hat * weight + bias
        return y
    
    def backward_0(self, grad):
        x, weight, bias = self.inputs
        N = x.shape[0]
        return (1. / N) * weight * (self.var + self.epsilon)**(-1. / 2.) \
                * (N * grad - np.sum(grad, axis=0) - (x - self.mean) * (self.var + self.epsilon)**(-1.0) * np.sum(grad * (x - self.mean), axis=0))

    def backward_1(self, grad):
        return grad * self.x_hat

    def backward_2(self, grad):
        return grad