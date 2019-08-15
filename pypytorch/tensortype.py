# -*- coding: utf-8 -*-


import numpy as np


class TensorType(object):

    def __init__(self, type):
        self.type = type

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.__class__).split("'")[1] + '.' + str(self.type).split("'")[0].split('.')[-1]

    def __eq__(self, other):
        return self.type == other.type


_int16 = np.int16
_int32 = np.int32
_int64 = np.int64

_float32 = np.float32

int16 = TensorType(_int16)
int32 = TensorType(_int32)
int64 = TensorType(_int64)

float32 = TensorType(_float32)
