# -*- coding: utf-8 -*-


import unittest
from unittest import TestCase
import pypytorch as ppt
import os


class ModuleTest(TestCase):


    @unittest.skip
    def test_sequential(self):
        class MyModule(ppt.nn.Module):

            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = ppt.nn.Sequential(
                    ppt.nn.Conv2d(3, 1, 2),
                    ppt.nn.MaxPool2d(2, 2)
                )
            
            def forward(self, x):
                return self.conv(x)
        
        model = MyModule()
        print(model)
    
    @unittest.skip
    def test_deconv2d(self):
        class MyModule(ppt.nn.Module):


            def __init__(self):
                super(MyModule, self).__init__()
                self.conv1 = ppt.nn.Conv2d(4, 1, 2)
                self.conv2 = ppt.nn.DeConv2d(1, 4, 2)
            
            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(conv1)
                return conv2
        
        x = ppt.np.arange(0, 4 * 4 * 4 * 4).reshape(4, 4, 4, 4).astype('float32')
        x = ppt.tensor(x)
        model = MyModule()
        print(model)
        out = model(x)
        out.backward(ppt.ones_like(out))
        print(out.grad)

    def test_nearest_upsample(self):
        class MyModule(ppt.nn.Module):


            def __init__(self):
                super(MyModule, self).__init__()
                self.conv1 = ppt.nn.Conv2d(4, 1, 2)
                self.conv2 = ppt.nn.NearestUpsample((6, 6))
            
            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(conv1)
                return conv2
        
        x = ppt.np.arange(0, 4 * 4 * 4 * 4).reshape(4, 4, 4, 4).astype('float32')
        x = ppt.tensor(x)
        model = MyModule()
        print(model)
        out = model(x)
        out.backward(ppt.ones_like(out))
        print(model.conv1.weight.grad)