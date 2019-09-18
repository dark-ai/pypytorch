# -*- coding: utf-8 -*-


import numpy as np
import pypytorch as ppt
from pypytorch.functions.function import Function


class BatchNorm(Function):


    def __init__(self, momentum=0.999):
        super(BatchNorm, self).__init__()
        self.mean = 0.0
        self.var = 0.0
        self.running_mean = 0.0
        self.running_var = 0.0
        self.epsilon = 1e-3
        self.momentum = momentum
        self.x_hat = 0.0

    def forward(self, x, gamma, beta):
        """
        Parameters
        ----------
        x : Tensor
            Input feature map
        
        gamma : float

        beta : float
        """
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0)
        self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.mean
        self.running_var = self.momentum * self.var + (1 - self.momentum) * self.var
        x_hat = (x - self.mean) / np.sqrt(self.var ** 2 + self.epsilon)
        y = x_hat * gamma + beta
        return y
    
    def backward_0(self, grad):
        x, gamma, beta = self.inputs
        batch, channel, height, width = x.shape
        d_var = np.sum(gamma * (x - self.mean), axis=0) * -1 / 2 * (self.var + self.epsilon) ** (-3 / 2)
        d_mean = np.sum(gamma * (-1 / np.sqrt(self.var + self.epsilon)), axis=0) \
                    + d_var * np.mean((-2 * (x - self.mean)), axis=0)
        return gamma * (1 / np.sqrt(self.var + self.epsilon)) + d_var * 2 * (x - self.mean) / batch + d_mean * 1 / batch

    def backward_1(self, grad):
        return self.x_hat.sum()

    def backward_2(self, grad):
        return 1.0