# -*- coding: utf-8 -*-

import math
import numpy as np

from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import utils
from pypytorch import functions


class MaxPool2d(Module):

    
    def __init__(self, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        return functions.max_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)

    def train(self):
        self.prepare_modules_for_train()
    
    def eval(self):
        self.prepare_modules_for_eval()

    def __str__(self):
        return 'MaxPool2d(kernel_size=%s, stride=%s, padding=%s)'\
            % (self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return str(self)


class AvgPool2d(Module):


    def __init__(self, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        return functions.avg_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)

    def train(self):
        self.prepare_modules_for_train()
    
    def eval(self):
        self.prepare_modules_for_eval()

    def __str__(self):
        return 'AvgPool2d(kernel_size=%s, stride=%s, padding=%s)'\
            % (self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return str(self)
