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
    def __init__(gamma, beta, momentum=0.999):
        super(BatchNorm, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
    
    def forward(self, x):
        return functions.batch_norm(x, self.gamma, self.beta, momentum=self.momentum)

    def __str__(self):
        return 'Batch(gamma=%s, beta=%s, momentum=%s)' \
            % (self.gamma, self.beta, self.momentum)

    def __repr__(self):
        return str(self)

    def train(self):
        self.prepare_modules_for_train()
        self.weight.requires_grad = True
        
    def eval(self):
        self.prepare_modules_for_eval()
        self.weight.requires_grad = False
