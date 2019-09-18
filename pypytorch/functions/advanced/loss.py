# -*- coding: utf-8 -*-

import numpy as np
from pypytorch.functions.function import Function
from pypytorch import utils


class MSELoss(Function):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.num = 0

    def forward(self, predicted, labels):
        self.num = len(labels)
        return ((predicted - labels) ** 2).sum() / self.num
    
    def backward_0(self, grad):
        predicted, labels = self.inputs
        return grad * ((predicted - labels) * 2) / self.num
    
    def backward_1(self, grad):
        predicted, labels = self.inputs
        return grad * ((predicted - labels) * -2) / self.num


class NLLLoss(Function):

    def __init__(self):
        super(NLLLoss, self).__init__()
        self.len = 0

    def forward(self, predicted, labels):
        """
        Parameters
        ----------
        labels : one-hot or indice
        """

        self.len = len(labels)
        return (-labels * predicted).sum() / self.len

    def backward_0(self, grad):
        predicted, labels = self.inputs
        return grad * -labels / self.len

    def backward_1(self, grad):
        predicted, labels = self.inputs
        return grad * -predicted / self.len


class Softmax(Function):

    def __init__(self):
        super(Softmax, self).__init__()
        self.output = 0.0

    def forward(self, x, dim):
        self.output = utils.softmax(x, dim)
        return self.output

    def backward_0(self, grad):
        x, dim = self.inputs
        out = self.output
        out_grad = np.zeros_like(grad)
        # row is a single row of grad
        for idx, row in enumerate(grad):
            sum_ = (row * out[idx]).sum()
            out_grad[idx] = out[idx] * (row - sum_)
        return out_grad
