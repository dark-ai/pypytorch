# -*- coding: utf-8 -*-


class Optimizer(object):


    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
