# -*- coding: utf-8 -*-

import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch.nn import functional as F


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
        self.train = train

    def forward(self, x):
        return F.dropout(x, prob=self.prob, train=self.train)

    def __str__(self):
        return 'DropOut(prob=%s, train=%s)' % (self.prob, self.train)

    def __repr__(self):
        return str(self)
