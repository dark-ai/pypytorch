# -*- coding: utf-8 -*-

from .functions import *
from pypytorch import utils


def linear(inputs, weight, bias):
    linear = Linear()
    return linear(inputs, weight, bias)

def conv2d(inputs, weight, bias, stride=(1, 1), padding=(0, 0)):
    stride, padding = utils.pair_tuple(stride, padding)
    conv = Conv2d(stride, padding)
    return conv(inputs, weight, bias)

def max_pool2d(inputs, kernel_size, stride=(1, 1), padding=(0, 0)):
    kernel_size, stride, padding = utils.pair_tuple(kernel_size, stride, padding)
    max_pool = MaxPool2d(stride, padding)
    return max_pool(inputs, kernel_size)

def avg_pool2d(inputs, kernel_size, stride=(1, 1), padding=(0, 0)):
    kernel_size, stride, padding = utils.pair_tuple(kernel_size, stride, padding)
    avg_pool = AvgPool2d(stride, padding)
    return avg_pool(inputs, kernel_size)

def relu(inputs):
    relu = ReLU()
    return relu(inputs)

def sigmoid(inputs):
    sigmoid = Sigmoid()
    return sigmoid(inputs)

def tanh(inputs):
    tanh = Tanh()
    return tanh(inputs)

def mse_loss(predicted, labels):
    mse_loss = MSELoss()
    return mse_loss(predicted, labels)

def softmax(inputs, dim=0):
    softmax = Softmax()
    return softmax(inputs, dim)

def nll_loss(predicted, labels):
    label_num = len(predicted[0])
    labels = utils.ensure_one_hot(labels, label_num)
    nll_loss = NLLLoss()
    return nll_loss(predicted, labels)

def cross_entropy_loss(predicted, labels):
    output = softmax(predicted, dim=1)
    return nll_loss(output.log(), labels)
