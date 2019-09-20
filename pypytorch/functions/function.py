# -*- coding: utf-8 -*-

from pypytorch.torchnode import GradFn
import pypytorch as t


def get_backward_names(o):
    names = []
    length = len(o.inputs)
    count = 0
    for prop in dir(o):
        if count >= length:
            return names
        if prop.startswith('backward_'):
            names.append(prop)
            count += 1
    return names


def backward(o, wrap):

    def backward_with_broadcast(grad):
        oprand = int(wrap.__name__.split('_')[1])
        ret = t.Tensor(o.broadcast(wrap(grad), oprand))
        return ret
    
    return backward_with_broadcast


class Function(object):

    
    def __init__(self):
        """
        Notes
        If need extra arguments in __init__() method when inhrent Function, note that these arguments are unlearnable,
        as for the learnable arguments, please go to Function.forward(self, *args)
        """
        self.inputs = []
        self.raw_inputs = []

    def __call__(self, *args):
        self.raw_inputs = list(args)
        for i, arg in enumerate(args):
            if arg is None:
                self.raw_inputs.pop(i)
        for arg in self.raw_inputs:
            if isinstance(arg, t.Tensor):
                self.inputs.append(arg.data)
            else:
                self.inputs.append(arg)
        data = self.forward(*self.inputs)
        requires_grad = any(list(map(lambda x: x.requires_grad if hasattr(x, 'requires_grad') else False, self.raw_inputs)))
        grad_fns = []
        backward_names = get_backward_names(self)
        for backward_name in backward_names:
            back_name, oprand = backward_name.split('_')
            oprand = int(oprand)
            # wrap grad_fn by broadcast
            fn = backward(self, getattr(self, backward_name))
            grad_fn = GradFn(self.__class__.__name__ + back_name[0].upper() + back_name[1:] + str(oprand),\
                self.raw_inputs[oprand], fn)
            self.raw_inputs[oprand].grad_fn = grad_fn
            grad_fns.append(grad_fn)
        output_tensor = t.Tensor(data, requires_grad=requires_grad)
        output_tensor._node.register_grad_fns(grad_fns)
        return output_tensor

    
    def forward(self, *args):
        """
        Parameters
        ----------
        args : tuple (built-in Python class or numpy.ndarray)
            The elements in tuple are learnable, like input, weight, bias in conv operator
        """
        raise NotImplementedError
    
    def broadcast(self, grad, oprand):
        ndim_added = grad.ndim - self.inputs[oprand].ndim
        # broadcast stage 1
        for i in range(ndim_added):
            grad = grad.sum(axis=0)
        
        # broadcast stage 2
        for i, dim in enumerate(grad.shape):
            if dim != self.inputs[oprand].shape[i]:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

