# -*- coding: utf-8 -*-

import pypytorch as t


class LeNetV1(t.nn.Module):


    def __init__(self, in_ch, out_ch):
        super(LeNetV1, self).__init__()
        self.conv1 = t.nn.Conv2d(in_ch, 4, 5, padding='SAME')
        self.pool1 = t.nn.MaxPool2d(2, 2)
        self.conv2 = t.nn.Conv2d(4, 8, 5, 2, padding='SAME')
        self.pool2 = t.nn.MaxPool2d(2, 2)
        self.fc1 = t.nn.Linear(288, out_ch)
    
    def forward(self, x):
        out = self.pool1(t.functions.relu(self.conv1(x)))
        out = self.pool2(t.functions.relu(self.conv2(out)))
        out = out.view((out.shape[0], -1))
        out = t.functions.relu(self.fc1(out))
        return out


class LeNetV2(t.nn.Module):


    def __init__(self, in_ch, out_ch):
        super(LeNetV2, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_ch, 4, 5, padding='SAME'),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, 2)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(4, 8, 5, 2, padding='SAME'),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, 2)
        )
        self.fc1 = t.nn.Linear(288, out_ch)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view((out.shape[0], -1))
        out = t.functions.relu(self.fc1(out))
        return out

class LeNetV3(t.nn.Module):


    def __init__(self, in_ch, out_ch):
        super(LeNetV3, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_ch, 4, 5, padding='SAME'),
            t.nn.Tanh(),
            t.nn.MaxPool2d(2, 2)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(4, 8, 5, 2, padding='SAME'),
            t.nn.Tanh(),
            t.nn.MaxPool2d(2, 2)
        )
        self.fc1 = t.nn.Linear(288, out_ch)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view((out.shape[0], -1))
        out = t.functions.tanh(self.fc1(out))
        return out


class LeNetV4(t.nn.Module):


    def __init__(self, in_ch, out_ch):
        super(LeNetV4, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(in_ch, 4, 5, padding='SAME'),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, 2)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(4, 8, 5, 2, padding='SAME'),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2, 2)
        )
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(288, 250),
            t.nn.DropOut(),
            t.nn.Linear(250, out_ch)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view((out.shape[0], -1))
        out = t.functions.relu(self.fc1(out))
        return out