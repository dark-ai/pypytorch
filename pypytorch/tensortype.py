# -*- coding: utf-8 -*-


import numpy as np


class TensorType(object):

    def __init__(self, type):
        self.type = type

    def __repr__(self):
        return str(self)

    def __str__(self):
        class_full_name = str(self.__class__).split("'")[1]
        moudle_name, class_short_name = '.'.join(class_full_name.split('.')[0:-1]), class_full_name.split('.')[-1]
        return moudle_name + '.' + str(self.type).split("'")[1].split('.')[-1]

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
