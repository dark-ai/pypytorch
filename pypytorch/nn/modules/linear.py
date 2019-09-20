# -*- coding: utf-8 -*-

import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import functions


class Linear(Module):
    """
    Notes
    -----
    In subclass of Module, don't define self._name b/c it's used in a special way
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Notes
        -----
        self.weight and self.bias store parameters to train
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = t.Tensor((in_features, out_features), requires_grad=True)
        self.bias = t.Tensor((1, out_features), requires_grad=True)
        if bias is None:
            self.bias = None
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[0])
        self.weight.data = np.random.uniform(-stdv, stdv, self.weight.data.shape)
        if self.bias.requires_grad:
            self.bias.data = np.random.uniform(-stdv, stdv, self.bias.data.shape)

    def train(self):
        self.prepare_modules_for_train()
        self.weight.requires_grad = True
        if hasattr(self, 'bias') and getattr(self, 'bias') is not None:
            getattr(self, 'bias').requires_grad = True
        
    def eval(self):
        self.prepare_modules_for_eval()
        self.weight.requires_grad = False
        if hasattr(self, 'bias') and getattr(self, 'bias') is not None:
            getattr(self, 'bias').requires_grad = False

    def forward(self, x):
        return functions.linear(x, self.weight, self.bias)

    def __str__(self):
        return 'Linear(in_features=%s, out_features=%s, bias=%s)' % (self.in_features, self.out_features, True if self.bias is not None else False)

    def __repr__(self):
        return str(self)