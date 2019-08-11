# -*- coding: utf-8 -*-

from pypytorch.nn.modules import Module
from pypytorch.nn import functional as F


class MSELoss(Module):

    def forward(self, predicted, labels):
        return F.mse_loss(predicted, labels)
    

class CrossEntropyLoss(Module):

    def forward(self, predicted, labels):
        return F.cross_entropy_loss(predicted, labels)


class NLLLoss(Module):
    
    def forward(self, predicted, labels):
        return F.nll_loss(predicted, labels)
