# -*- coding: utf-8 -*-


import torch as t
import torch.nn as nn
import torch.nn.functional as F


class LeNetV1(nn.Module):


    def __init__(self, in_ch, out_ch):
        super(LeNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 4, 5, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 5, 2, padding=7),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(288, 250),
            nn.Dropout(),
            nn.Linear(250, out_ch)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        return out
