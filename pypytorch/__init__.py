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
from pypytorch.data import transform
from pypytorch import optim
from pypytorch import functions
from pypytorch import utils
from pypytorch.tensor import tensor

# third party modules
import numpy as np
import matplotlib.pyplot as plt
import dill


def ignore_ellipsis(flag):
    if flag:
        np.set_printoptions(threshold=np.inf)
    else:
        np.set_printoptions(threshold=1000)


def save(model, fname):
    with open(fname, 'wb') as f:
        dill.dump(model, f)

def load(fname):
    with open(fname, 'rb') as f:
        return dill.load(f)

def ones(shape):
    return Tensor(np.ones(shape), dtype=int64)

def zeros(shape):
    return Tensor(np.zeros(shape), dtype=int64)

def ones_like(tensor):
    return Tensor(np.ones_like(tensor.data), dtype=int64)

def zeros_like(tensor):
    return Tensor(np.zeros_like(tensor.data), dtype=int64)

__version__ = 'v0.0.1'
