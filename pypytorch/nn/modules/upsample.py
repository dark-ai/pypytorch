# -*- coding: utf-8 -*-

import numpy as np

from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch import utils
from pypytorch import functions


class NearestUpsample(Module):

    
    def __init__(self, size):
        super(NearestUpsample, self).__init__()
        self.size = size
        
    def forward(self, x):
        return functions.nearest_upsample(x, self.size)

    def __str__(self):
        return 'NearestUpsample(size=(%s, %s))' % self.size

    def __repr__(self):
        return str(self)
