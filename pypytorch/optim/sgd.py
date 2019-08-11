# -*- coding: utf-8 -*-

from collections import Iterable, Iterator
from pypytorch.optim.optimizer import Optimizer


class SGD(Optimizer):

    
    def step(self):
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad.data