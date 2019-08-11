# -*- coding: utf-8 -*-

from collections import Iterable
from collections import Iterator
from collections import OrderedDict


def extract_modules(o):
    named_moudles = OrderedDict()
    for key in o.__dict__:
        if key in ['weight', 'bias']:
            continue
        value = o.__dict__[key]
        if isinstance(value, Module):
            named_moudles[key] = value
    return named_moudles


class Module(object):
    

    def __init__(self):
        self._name = self.__class__.__name__
        self._modules = []
        self._named_modules = OrderedDict()
        self._parameters = []
        self._named_parameters = OrderedDict()
        self.training = True

    @property
    def modules(self):
        assert hasattr(self, '_modules'), 'should call super(Class, self).__init__() in __init__'
        # self._modules = self._modules if self._modules else tuple(extract_modules(self))
        self._named_modules = self._named_modules if self._named_modules else extract_modules(self)
        self._modules = self._named_modules.values()
        return self._modules
    
    def named_modules(self):
        assert hasattr(self, '_named_modules'), 'should call super(Class, self).__init__() in __init__'
        self._modules
        return self._named_modules

    def eval(self):
        self.training = False
        self.modules
        for param in self.parameters():
            param.requires_grad = False
    
    def train(self):
        self.training = True
        self.modules
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        assert hasattr(self, '_modules'), 'module must inherit pypytorch.nn.module.Module'
        self._modules = tuple(extract_modules(self))
        return self.forward(*args, **kwargs)
    
    @property
    def name(self):
        return self._name    

    def parameters(self):
        self.modules
        
        if self._parameters:
            return self._parameters

        if hasattr(self, 'weight') and self.weight is not None:
            self._parameters.append(self.weight)

        if hasattr(self, 'bias') and self.bias is not None:
            self._parameters.append(self.bias)
        
        for module in self._modules:
            self._parameters.extend(module.parameters())
        return self._parameters

    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def _description(self, num_space=0):
        self.modules
        indentation = ' ' * 2
        space = ' ' * num_space
        s = self._name + '(\n'
        for key, value in self.named_modules().items():
            value_str = str(value) if not value.modules else value._description(num_space=num_space * 2 if num_space else 2)
            s += space + indentation + '(' + key + '): ' + value_str + '\n'
        s += space + ')'
        return s

    def __str__(self):
        return self._description()

    def __repr__(self):
        return str(self)
