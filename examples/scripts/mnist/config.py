# -*- coding: utf-8 -*-

import warnings


class Config(object):

    data_dir = '../../data/mnist/'
    model = 'LeNetV2'
    model_path = './checkpoints/LeNetV2/2019-08-15_21-27-23_1.pth'
    print_seq = 20
    lr = 0.1
    batch_size = 32
    epochs = 10
    lr_decay = 0.95

    def parse_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn("Invalid option `%s`" % key)

