# -*- coding: utf-8 -*-

import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import functions


class ReLU(Module):
    """
    Notes
    -----
    In subclass of Module, don't define self._name b/c it's used in a special way
    """
    def __init__(self):
        """
        Notes
        -----
        self.weight and self.bias store parameters to train
        """
        super(ReLU, self).__init__()

    def forward(self, x):
        return functions.relu(x)

    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()

    def __str__(self):
        return 'ReLU()'

    def __repr__(self):
        return str(self)


class Sigmoid(Module):


    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, x):
        return functions.sigmoid(x)
    
    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()

    def __str__(self):
        return 'Sigmoid()'
    
    def __repr__(self):
        return str(self)


class Tanh(Module):


    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, x):
        return functions.tanh(x)

    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()

    def __str__(self):
        return 'Tanh()'
    
    def __repr__(self):
        return str(self)
