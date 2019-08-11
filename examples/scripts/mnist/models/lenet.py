# -*- coding: utf-8 -*-

import pypytorch as t

from models import BaseModule


class LeNet(BaseModule):


    def __init__(self, in_ch, out_ch):
        super(LeNet, self).__init__()
        self.conv1 = t.nn.Conv2d(in_ch, 4, 5, padding='SAME')
        self.pool1 = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(4, 8, 5, 2, padding='SAME')
        self.pool2 = t.nn.MaxPool2d(2, 2)
        self.fc1 = t.nn.Linear(288, out_ch)
    
    def forward(self, x):
        out = self.pool1(t.F.relu(self.conv1(x)))
        out = self.pool2(t.F.relu(self.conv2(out)))
        out = out.view((out.shape[0], -1))
        out = t.F.relu(self.fc1(out))
        return out
