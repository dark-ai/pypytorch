# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import math
import numpy as np
from pypytorch.nn.modules.module import Module
import pypytorch as t
from pypytorch.nn import functional as F


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
        return F.relu(x)

    def __str__(self):
        return 'ReLU()'

    def __repr__(self):
        return str(self)