# -*- coding: utf-8 -*-

import pypytorch as ppt
import pypytorch.nn as nn
from pypytorch.nn import Module
import time


class VGG16(Module):


    def __init__(self, in_ch, classes):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='SAME'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='SAME'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='SAME'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='SAME'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block6 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view((out.shape[0], -1))
        out = self.block6(out)
        return out