# -*- coding: utf-8 -*-


import numpy as np
import pypytorch as t


class Transformer(object):

    def __call__(self, data):
        return self.transform(data)

    def transform(self, data):
        raise NotImplementedError


class Compose(Transformer):

    def __init__(self, *transformers):
        self.transformers = transformers
    
    def transform(self, data):
        for trans in self.transformers:
            data = trans(data)
        return data


class ToTensor(Transformer):


    def transform(self, data):
        """
        Parameters
        ----------
        data : PIL.Image.open function returned value
            format is [height, width, channels]
        """
        data = np.array(data)
        if data.ndim == 2:
            data = data.reshape((-1, 1) + data.shape)
        if data.ndim == 3:
            data = data.reshape((-1, ) + data.shape)
            data = np.transpose(data, (0, 3, 1, 2))
        return t.Tensor(data)


class Norm(Transformer):


    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        """
        Parameters
        ----------
        data : tensor which doesn't require grad
        """
        assert not data.requires_grad, "data should't requires_grad"
        assert len(self.mean) == data.shape[1], "dims of mean doesn't match data"
        assert len(self.std) == data.shape[1], "dims of std doesn't match std"
        data.data = data.data / 127.5 - 1
        for i in range(len(self.mean)):
            data.data[:, i] = (data.data[:, i] - self.mean[i]) / self.std[i]
        return data
