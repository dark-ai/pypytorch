# -*- coding: utf-8 -*-

import warnings


class Config(object):

    data_dir = '../../data/cifar10/'
    model = 'VGG16'
    model_path = None
    print_seq = 1
    lr = 0.1
    batch_size = 8
    epochs = 10
    lr_decay = 0.95

    def parse_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn("Invalid option `%s`" % key)

