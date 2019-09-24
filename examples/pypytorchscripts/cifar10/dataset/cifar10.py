# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from PIL import Image
import pypytorch as ppt
from pypytorch.data import Dataset
import pypytorch.data.transform as transform


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        X = data['data']
        y = data['labels']
        # X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        return X, y


class Cifar10(Dataset):


    def __init__(self, root, filenames, train=True):
        super(Cifar10, self).__init__()
        X, y = None, None
        for fname in filenames:
            path = os.path.join(root, fname)
            if X is not None and y is not None:
                X_new, y_new = load_data(path)
                X = np.concatenate([X, X_new], axis=0)
                y = np.concatenate([y, y_new], axis=0)
            else:
                X, y = load_data(path)
        self.transforms = transform.Compose(
            transform.Resize(224),
            transform.ToTensor(),
            # transform.Norm()
        )
        self.imgs = X.reshape(len(X), 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = y
    
    def __getitem__(self, idx):
        img = self.transforms(self.imgs[idx])
        label = ppt.utils.ensure_one_hot(self.labels[idx], 10)
        return img, label

    def __len__(self):
        return len(self.imgs)


def main():
    root = '../../../data/cifar10/'
    cifar10 = Cifar10(root, \
        ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
    x, y = cifar10[0]
    img = transform.ToPILImage()(x)
    img.show()


if __name__ == "__main__":
    main()
