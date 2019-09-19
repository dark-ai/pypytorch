# -*- coding: utf-8 -*-


import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import functions


class BatchNorm2d(Module):
    """
    Notes
    -----
    In subclass of Module, don't define self._name b/c it's used in a special way
    """
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.1, train=True):
        super(BatchNorm2d, self).__init__()
        self.gamma = t.Tensor(gamma, requires_grad=True)
        self.beta = t.Tensor(beta, requires_grad=True)
        self.momentum = momentum
        self.training = train
    
    def forward(self, x):
        return functions.batch_norm2d(x, self.gamma, self.beta, momentum=self.momentum, train=self.training)

    def __str__(self):
        return 'BatchNorm2d(momentum=%s, train=%s)' % (self.momentum, self.training)

    def __repr__(self):
        return str(self)

    def train(self):
        self.prepare_modules_for_train()
        self.gamma.requires_grad = True
        self.beta.requires_grad = True
        
    def eval(self):
        self.prepare_modules_for_eval()
        self.gamma.requires_grad = False
        self.beta.requires_grad = False
