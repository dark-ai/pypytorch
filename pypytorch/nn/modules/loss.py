# -*- coding: utf-8 -*-

from pypytorch.nn.modules import Module
from pypytorch import functions


class MSELoss(Module):

    def forward(self, predicted, labels):
        return functions.mse_loss(predicted, labels)
    
    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()


class CrossEntropyLoss(Module):

    def forward(self, predicted, labels):
        return functions.cross_entropy_loss(predicted, labels)
    
    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()


class NLLLoss(Module):
    
    def forward(self, predicted, labels):
        return functions.nll_loss(predicted, labels)

    def eval(self):
        self.prepare_modules_for_eval()
    
    def train(self):
        self.prepare_modules_for_train()
