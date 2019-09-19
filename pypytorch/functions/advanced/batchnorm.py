# -*- coding: utf-8 -*-


import numpy as np
import pypytorch as ppt
from pypytorch.functions.function import Function


class BatchNorm2d(Function):


    def __init__(self, momentum=0.1, epsilon=1e-5, train=True):
        super(BatchNorm2d, self).__init__()
        self.mean = 0.0
        self.var = 0.0
        self.running_mean = 0.0
        self.running_var = 0.0
        self.momentum = momentum
        self.epsilon = epsilon
        self.x_hat = 0.0
        self.train = train

    def forward(self, x, gamma, beta):
        if self.train:
            self.mean = x.mean(axis=0)
            self.var = x.var(axis=0)
            self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.var + (1 - self.momentum) * self.var
            self.x_hat = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.var ** 2 + self.epsilon)
        y = self.x_hat * gamma + beta
        return y
    
    def backward_0(self, grad):
        x, gamma, beta = self.inputs
        batch, channel, height, width = x.shape
        d_var = np.sum(gamma * (x - self.mean), axis=0) * -1 / 2 * (self.var + self.epsilon) ** (-3 / 2)
        d_mean = np.sum(gamma * (-1 / np.sqrt(self.var + self.epsilon)), axis=0) \
                    + d_var * np.mean((-2 * (x - self.mean)), axis=0)
        return grad * gamma * (1 / np.sqrt(self.var + self.epsilon)) + d_var * 2 * (x - self.mean) / batch + d_mean * 1 / batch

    def backward_1(self, grad):
        return grad * self.x_hat.sum()

    def backward_2(self, grad):
        return grad