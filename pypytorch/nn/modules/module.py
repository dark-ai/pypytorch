# -*- coding: utf-8 -*-

import os
import time
from collections import OrderedDict
import json
import pypytorch as t


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
        self._modules = list(self._named_modules.values())
        return self._modules
    
    def named_modules(self):
        assert hasattr(self, '_named_modules'), 'should call super(Class, self).__init__() in __init__'
        self._modules
        return self._named_modules

    def prepare_modules_for_train(self):
        self.training = True
        self.prepare_modules()
    
    def prepare_modules_for_eval(self):
        self.training = False
        self.prepare_modules()

    def prepare_modules(self):
        self.modules

    def eval(self):
        self.prepare_modules_for_eval()
        for module in self.modules:
            module.eval()

    def train(self):
        self.prepare_modules_for_train()
        for module in self.modules:
            module.train()

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

        if hasattr(self, 'weight') and getattr(self, 'weight') is not None:
            self._parameters.append(getattr(self, 'weight'))

        if hasattr(self, 'bias') and getattr(self, 'bias') is not None:
            self._parameters.append(getattr(self, 'bias'))
        
        for module in self._modules:
            self._parameters.extend(module.parameters())
        return self._parameters

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def save(self, epoch, loss, max_item=5, root='checkpoints/'):
        assert max_item > 0, 'max_item must be gt 0'
        model_dir = os.path.join(root, self.name)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        weight_path = os.path.join(model_dir, current_time + '_' + str(epoch) + '.pth')
        checkpoints_path = os.path.join(model_dir, 'checkpoints.json')
        if not os.path.exists(checkpoints_path):
            with open(checkpoints_path, 'w') as f:
                f.write('{}')
        with open(checkpoints_path, 'r') as fr:
            data = json.load(fr)
        if len(data) == max_item:
            data = sorted(data.items(), key=lambda x: x[1])
            name = data.pop()[0]
            full_name = os.path.join(model_dir, name)
            os.remove(full_name)
            data = dict(data)
            with open(checkpoints_path, 'w') as fw:
                json.dump(data, fw, indent=4)
            
        with open(checkpoints_path, 'r') as f:
            data = json.load(f)
        data = dict(sorted(data.items(), key=lambda x: x[1]))
        t.save(self, weight_path)
        # print(loss.data.tolist())
        data[current_time + '_' + str(epoch) + '.pth'] = loss.data.tolist()
        with open(checkpoints_path, 'w') as f:
            json.dump(data, f, indent=4)
    
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


class Sequential(Module):


    def __init__(self, *modules):
        super(Sequential, self).__init__()
        assert len(modules) != 0, "At least need one module"
        for i, mod in enumerate(modules):
            setattr(self, str(i), mod)

    def forward(self, x):
        out = x
        for mod in self.modules:
            out = mod(out)
        return out
