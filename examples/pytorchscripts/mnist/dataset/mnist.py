# -*- coding: utf-8 -*-


import os
from PIL import Image
import torch as t
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms as T


class Mnist(Dataset):


    def __init__(self, root):
        self.root = root
        self.imgs = os.listdir(root)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.3], std=[0.5])
        ])
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(path)
        x = self.transforms(img)
        label = int(self.imgs[idx].split('.')[0])
        return x, label
    
    def __len__(self):
        return len(self.imgs)
