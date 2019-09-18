# -*- coding: utf-8 -*-

from pypytorch.functions.function import Function


class Eq(Function):


    def forward(self, a, b):
        return (a == b).astype('int32')
    
    def backward_0(self, grad):
        return grad


class Lt(Function):


    def forward(self, a, b):
        return (a < b).astype('int32')
    
    def backward_0(self, grad):
        return grad


class Gt(Function):


    def forward(self, a, b):
        return (a > b).astype('int32')
    
    def backward_0(self, grad):
        return grad


class Le(Function):


    def forward(self, a, b):
        return (a <= b).astype('int32')
    
    def backward_0(self, grad):
        return grad


class Ge(Function):


    def forward(self, a, b):
        return (a >= b).astype('int32')
    
    def backward_0(self, grad):
        return grad


class Ne(Function):


    def forward(self, a, b):
        return (a != b).astype('int32')
    
    def backward_0(self, grad):
        return grad
