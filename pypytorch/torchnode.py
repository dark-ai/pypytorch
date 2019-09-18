# -*- coding: utf-8 -*-


import pypytorch as t


class GradFn(object):


    def __init__(self, name, tensor, fn):
        self._name = name
        self._tensor = tensor
        self.fn = fn

    @property
    def name(self):
        return self._name

    @property
    def tensor(self):
        return self._tensor

    def __call__(self, grad):
        return self.fn(grad)

    def __str__(self):
        return '<%s at %s>' % (self.name, str(self.fn).split(' ')[-1][:-1])
    
    def __repr__(self):
        return str(self)


class TorchNode(object):


    def __init__(self, tensor=None, requires_grad=False, grad_fns=None):
        self.tensor = tensor
        self.requires_grad = requires_grad
        self.grad_fns = grad_fns or []
    
    def register_grad_fn(self, grad_fn):
        self.grad_fns.append(grad_fn)
    
    def register_grad_fns(self, grad_fns):
        self.grad_fns.extend(grad_fns)
    
    def backward(self, grad=None):
        """
        Parameters
        ----------
        grad : Tensor
        """
        tensor = self.tensor
        if grad is None:
            if self.tensor.shape == ():
                grad = t.Tensor([1.0])
            else:
                assert False, 'need input a grad tensor when tensor is not a scalar'
        if tensor.requires_grad:
            tensor.grad.data = tensor.grad.data + grad.data
        for grad_fn in tensor._node.grad_fns:
            if t.debug.BACKWARD_TRACE:
                print(grad_fn)
            output_grad = grad_fn(grad.data)
            grad_fn.tensor.backward(output_grad)
        
