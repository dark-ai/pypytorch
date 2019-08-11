# -*- coding: utf-8 -*-

import os
import time
import warnings
import pypytorch as t


class BaseModule(t.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    def save(self, root='checkpoints/'):
        model_dir = os.path.join(root, self.name)
        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        path = os.path.join(model_dir, current_time + '.pth')
        t.save(self, path)
    