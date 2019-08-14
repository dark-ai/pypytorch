#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_DIR)
import fire
import pypytorch as t

from mnist.config import Config
from mnist import models
from mnist.dataset import Mnist


opts = Config()
model = getattr(models, 'LeNetV1')(1, 10)

print(model)
def train(**kwargs):
    opts.parse_args(**kwargs)
    mnist = Mnist(opts.data_dir)
    optimizer = t.optim.SGD(model.parameters(), lr=opts.lr)
    criterion = t.nn.CrossEntropyLoss()
    dataloader = t.data.DataLoader(mnist, batch_size=opts.batch_size)

    for epoch in range(opts.epochs):
        avg_loss = 0.0
        for i, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted = model(data)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss
            if (i + 1) % opts.print_seq == 0:
                print('iteration: %s, loss: %s' % (i + 1, loss.data))
        avg_loss = avg_loss / i
        print('====epoch: %s, avg_loss: %s====' % (epoch + 1, avg_loss))
        t.utils.adjust_lr(optimizer, epoch + 1, opts.lr_decay, opts.learning_rate)
        model.save()
        


def main():
    fire.Fire()


if __name__ == "__main__":
    main()
