# -*- coding: utf-8 -*-

import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import functions


class DropOut(Module):
    """
    Notes
    -----
    In subclass of Module, don't define self._name b/c it's used in a special way
    """
    def __init__(self, prob=0.5, train=True):
        """
        Notes
        -----
        self.weight and self.bias store parameters to train
        """
        super(DropOut, self).__init__()
        self.prob = prob
        # training attribute is in super class(Moudle)
        self.training = train

    def forward(self, x):
        return functions.dropout(x, prob=self.prob, train=self.training)

    def train(self):
        self.prepare_modules_for_train()
    
    def eval(self):
        self.prepare_modules_for_eval()

    def __str__(self):
        return 'DropOut(prob=%s, train=%s)' % (self.prob, self.training)

    def __repr__(self):
        return str(self)
