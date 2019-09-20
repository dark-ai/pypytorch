# -*- coding: utf-8 -*-

import numpy as np

import pypytorch.tensortype as tt
from pypytorch.torchnode import GradFn
from pypytorch.torchnode import TorchNode
from pypytorch.functions import *
from pypytorch import utils


def tensor(data, dtype=tt.float32, mu=0.0, sigma=1.0):
    return Tensor(data, dtype=dtype, mu=mu, sigma=sigma)


class Tensor(object):


    def __init__(self, data, requires_grad=False, dtype=tt.float32, mu=0.0, sigma=1.0):
        """
        Notes
        -----
        When data is tuple, Tensor will init data by Gaussian distribution where mu=0.0
        sigma=1.0 by default
        """
        # TorchNode will run backward
        self._node = TorchNode(tensor=self, requires_grad=requires_grad)
        if isinstance(data, tuple):
            self._data = np.random.normal(loc=mu, scale=sigma, size=data).astype(dtype.type)
        else:
            self._data = np.array(data, dtype=dtype.type)
        self._dtype = dtype
        # self._node.requries_grad == False to avoid recurision
        self._grad = None
        self.zero_grad()
        self._grad_fn = None
        self._shape = self.data.shape

    @property
    def shape(self):
        return self._shape
    
    def numpy(self):
        return self.data
    
    def log(self):
        return Log()(self)

    def max(self, dim=None, keepdims=False):
        return Max()(self, dim, keepdims)
    
    def min(self, dim=None, keepdims=False):
        return Min()(self, dim, keepdims)

    def zeros_(self):
        self.data = np.zeros_like(self.data)
        return self
    
    def ones_(self):
        self.data = np.ones_like(self.data)
        return self

    def float_(self):
        self.dtype = tt.float32
        return self
    
    def long_(self):
        self.dtype = tt.int64
        return self

    def dim(self):
        return self._data.ndim
    
    def mean(self, dim=None, keepdims=False):
        return Mean()(self, dim, keepdims)
    
    def __len__(self):
        return len(self._data)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self.data = self.data.astype(dtype.type)

    @property
    def grad_fn(self):
        return self._grad_fn
    
    @grad_fn.setter
    def grad_fn(self, grad_fn):
        self._grad_fn = grad_fn

    def add(self, other):
        return Add()(self, utils.ensure_tensor(other))
    
    def sub(self, other):
        return Add()(self, Neg()(utils.ensure_tensor(other)))

    def sum(self, axis=None):
        return Sum()(self, axis)
    
    def mul(self, other):
        return Mul()(self, utils.ensure_tensor(other))
    
    def neg(self):
        return Neg()(self)
    
    def mm(self, other):
        return MM()(self, utils.ensure_tensor(other))

    def pow(self, other):
        return Pow()(self, utils.ensure_tensor(other))
    
    def div(self, other):
        return Div()(self, utils.ensure_tensor(other))
    
    def t(self, dims=None):
        if isinstance(dims, Tensor):
            dims = dims.data
        return Transpose()(self, dims)

    def view(self, size):
        if isinstance(size, Tensor):
            size = size.data
        return View()(self, size)
    
    def cat(self, other, dim=0):
        if isinstance(dim, Tensor):
            dim = dim.data
        return Concat()(self, utils.ensure_tensor(other), dim)
    
    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return utils.ensure_tensor(other).add(self)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __rsub__(self, other):
        return utils.ensure_tensor(other).sub(self)

    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return utils.ensure_tensor(other).mul(self)

    def __neg__(self):
        return self.neg()
    
    def __matmul__(self, other):
        return self.mm(other)
    
    def __getitem__(self, idxs):
        return Index()(self, idxs)
    
    def __pow__(self, other):
        return self.pow(other)
    
    def __truediv__(self, other):
        return self.div(other)
    
    def __eq__(self, other):
        return Eq()(self, utils.ensure_tensor(other))

    def __ne__(self, other):
        return Neg()(self, utils.ensure_tensor(other))
    
    def __lt__(self, other):
        return Lt()(self, utils.ensure_tensor(other))
    
    def __le__(self, other):
        return Le()(self, utils.ensure_tensor(other))
    
    def __gt__(self, other):
        return Gt()(self, utils.ensure_tensor(other))
    
    def __ge__(self, other):
        return Ge()(self, utils.ensure_tensor(other))

    @property
    def requires_grad(self):
        return self._node.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self._node.requires_grad = requires_grad
        if requires_grad:
            self.zero_grad()
        else:
            self.grad = None

    @property        
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
        self._shape = data.shape
        self._dtype.type = data.dtype
        self.zero_grad()

    @property    
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self._grad = grad

    def zero_grad(self):
        if self.requires_grad:
            self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad=None):
        """
        Parameters
        ----------
        grad : Tensor
        """
        if not self.requires_grad:
            return grad
        assert self.dtype == tt.float32, 'only float type can process backward'
        self._node.backward(grad)

    def __repr__(self):
        return str(self)

    def __str__(self):
        data_str = repr(self.data).replace('array(', '').replace(')', '')
        return 'Tensor(%s%s)'\
            % (data_str,
            ', grad_fn=<' + self.grad_fn.name + '>' if self.grad_fn else '')
    
    def __hash__(self):
        return id(self)

