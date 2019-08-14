# -*- coding: utf-8 -*-

from pypytorch.optim.optimizer import Optimizer


class SGD(Optimizer):

    
    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue
            param.data = param.data - self.lr * param.grad.data