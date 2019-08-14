# -*- coding: utf-8 -*-


import numpy as np
from functools import reduce
import pypytorch as t


class Dataset(object):

    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    

class DataLoader(object):


    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.len = len(dataset)
        self.iterated_num = 0
        self.start = 0
        self.batch_size = batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterated_num * self.batch_size < self.len:
            if self.start + self.batch_size > self.len:
                next_start = 0
                end = self.len
            else:
                next_start = end = self.start + self.batch_size
            data = None
            labels = None
            for i in range(self.start, end):
                if data is None:
                    data, labels = self.dataset[i][0].data, self.dataset[i][1].data
                else:
                    data = np.concatenate((data, self.dataset[i][0].data), axis=0)
                    labels = np.concatenate((labels, self.dataset[i][1].data), axis=0)
            data = t.Tensor(data)
            labels = t.Tensor(labels)
            self.start = next_start
            self.iterated_num += 1
            return data, labels
        else:
            self.iterated_num = 0
            raise StopIteration
