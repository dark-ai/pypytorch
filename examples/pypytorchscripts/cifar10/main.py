
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)
import time
import fire
import pypytorch as ppt
from pypytorch.data import DataLoader

from cifar10.config import Config
from cifar10 import models
from cifar10.dataset import Cifar10


opts = Config()
model = getattr(models, opts.model)(3, 10)

if opts.model_path:
    model = ppt.load(opts.model_path)
    print('Load weights...')
print(model)
model.train()


def train(**kwargs):
    opts.parse_args(**kwargs)
    cifar10 = Cifar10(opts.data_dir, \
        ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
    optimizer = ppt.optim.SGD(model.parameters(), lr=opts.lr)
    # optimizer = t.optim.Adam(model.parameters())
    criterion = ppt.nn.CrossEntropyLoss()
    dataloader = DataLoader(cifar10, batch_size=opts.batch_size)
    for epoch in range(opts.epochs):
        avg_loss = 0.0
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            start = time.time()
            predicted = model(data)
            end = time.time()
            print('Forward used: %ss' % (end - start))
            loss = criterion(predicted, labels)
            start = time.time()
            loss.backward()
            end = time.time()
            print('Backward used: %ss' % (end - start))
            optimizer.step()
            avg_loss += loss
            if (i + 1) % opts.print_seq == 0:
                print('Iteration: %s, loss: %s' % (i + 1, loss.data))
        avg_loss = avg_loss / i
        print('----Epoch: %s, avg_loss: %s----' % (epoch + 1, avg_loss))
        ppt.utils.adjust_lr(optimizer, epoch + 1, opts.lr, opts.lr_decay)
        model.save(epoch + 1, loss)
        print('Save weight')


def main():
    fire.Fire()

if __name__ == "__main__":
    main()
