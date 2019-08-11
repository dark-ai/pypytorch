# -*- coding: utf-8 -*-

import numpy as np
import pypytorch as t

def ensure_tensor(data):
    if isinstance(data, t.Tensor):
        return data
    return t.Tensor(data)

def ensure_tuple_list(data):
    if isinstance(data, tuple) or isinstance(data, list):
        return data
    return (data, data)

def ensure_one_hot(data, label_num):
    data = ensure_tensor(data)
    if data.dim() == 2:
        return data
    data.data = one_hot(data.data, label_num)
    return data

def one_hot(data, label_num):
    if data.shape == ():
        out = np.zeros((1, label_num))
        out[0, int(data.tolist())] = 1
        return out

    out = np.zeros((len(data), label_num))
    for i in range(len(data)):
        out[i, int(data[i])] = 1
    return out

def pair(data):
    return ensure_tuple_list(data)

def pair_tuple(*data):
    return map(lambda x: pair(x), data)

def make_padding(x, padding):
    return np.pad(x,
                    ((0, 0), (0, 0), 
                    (padding[0], padding[0]), 
                    (padding[1], padding[1])), 'constant', constant_values=0)

def unwrap_padding(x, padding):
    if padding == (0, 0):
        return x
    return x[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

def softmax(inputs, dim=0):
    exp = np.exp(inputs - np.max(inputs, axis=dim, keepdims=True))
    return exp / np.sum(exp, axis=dim, keepdims=True)

def adjust_lr(optimizer, epoch, initial_lr, lr_decay):
    lr = initial_lr / (1.0 + epoch * lr_decay)
    optimizer.lr = lr
    return lr