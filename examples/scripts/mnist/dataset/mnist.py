# -*- coding: utf-8 -*-


import os
import pypytorch as t
from PIL import Image


class Mnist(t.data.Dataset):


    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(root)
        self.transforms = t.transform.Compose(
            t.transform.ToTensor(),
            t.transform.Norm(mean=[0.3], std=[0.5])
        )
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(path)
        x = self.transforms(img)
        label = int(self.imgs[idx].split('.')[0])
        return x, t.utils.ensure_one_hot(label, 10)
    
    def __len__(self):
        return len(self.imgs)
