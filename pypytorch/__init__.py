# -*- coding: utf-8 -*-

import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)

from pypytorch.tensor import Tensor
from pypytorch.tensortype import *
from pypytorch import debug
from pypytorch import nn
from pypytorch import data
from pypytorch.data import dataset
from pypytorch.data import transform as T
from pypytorch import optim
from pypytorch import functions
from pypytorch.nn import functional as F
from pypytorch import utils

# third party modules
import numpy as np
import matplotlib.pyplot as plt
import dill


def save(fname, model):
    with open(fname, 'wb') as f:
        dill.dump(model, f, 1)

def load(fname):
    with open(fname, 'rb') as f:
        return dill.load(f)

__version__ = 'v0.0.1'
