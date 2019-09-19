# -*- coding: utf-8 -*-


import unittest
from unittest import TestCase
import pypytorch as t


class ModuleTest(TestCase):


    @unittest.skip
    def test_sequential(self):
        class MyModule(t.nn.Module):

            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = t.nn.Sequential(
                    t.nn.Conv2d(3, 1, 2),
                    t.nn.MaxPool2d(2, 2)
                )
            
            def forward(self, x):
                return self.conv(x)
        
        model = MyModule()
        print(model)
    