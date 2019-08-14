# -*- coding: utf-8 -*-

from pypytorch.optim.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(parameters, lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0.0
        self.v = 0.0
        self.t = 1
        self.state = {}
    
    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue
            if param not in self.state:
                self.state[param] = {
                    'm': self.m,
                    'v': self.v,
                    't': self.t,
                    'lr': self.lr
                }
                m = self.m
                v = self.v
                t = self.t
                lr = self.lr
            else:
                hyperparams = self.state[param]
                m = hyperparams['m']
                v = hyperparams['v']
                t = hyperparams['t']
                lr = hyperparams['lr']
            t += 1
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            m = self.beta1 * m + (1 - self.beta1) * param.grad.data
            v = self.beta2 * v + (1 - self.beta2) * (param.grad.data * param.grad.data)
            param.data = param.data - lr * m_hat / (v_hat ** 0.5 + self.epsilon)
            self.state[param] = {
                'm': m,
                'v': v,
                't': t,
                'lr': lr
            }
