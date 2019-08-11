# -*- coding: utf-8 -*-

import warnings


class Config(object):

    data_dir = '../../data/mnist/'
    model = 'LeNet'
    model_path = None
    print_seq = 20
    lr = 0.1
    batch_size = 32
    epochs = 100
    lr_decay = 0.95

    def parse_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn("Invalid option `%s`" % key)
