# -*- coding: utf-8 -*-


import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import functions


class BatchNorm(Module):
    """
    Notes
    -----
    In subclass of Module, don't define self._name b/c it's used in a special way
    """
    def __init__(self, mu=0.0, sigma=1.0, momentum=0.999):
        super(BatchNorm, self).__init__()
        self.gamma = t.Tensor(np.random.normal(loc=mu, scale=sigma, size=1), requires_grad=True)
        self.beta = t.Tensor(np.random.normal(loc=mu, scale=sigma, size=1), requires_grad=True)
        self.momentum = momentum
    
    def forward(self, x):
        return functions.batch_norm(x, self.gamma, self.beta, momentum=self.momentum)

    def __str__(self):
        return 'Batch(momentum=%s)' % (self.momentum)

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
