import numpy as np
# import h5py
import torch
# import glob
# import os
from torch.utils.data import Dataset
import json
from PIL import Image


class samho_data(Dataset):
    def __init__(self, data_type, transform=None, in_memory=False):
        if data_type == "train":
            self.ann_path = './data/train.json'
            self.data_path = './data/train'
        elif data_type == "test":
            self.ann_path = './data/test.json'
            self.data_path = './data/test'
        
        fd = open(self.ann_path)
        self.ann = json.load(fd)
        self._transform = transform

        if in_memory:
            self.data = torch.zeros(len(self.ann), 3, 512, 512)
            for i in range(len(self.ann)):
                pth = self.data_path + '/' + self.ann[i]['id']
                im = Image.open(pth).convert('RGB')
                newsize = (512, 512)
                im = im.resize(newsize)
                get_data = torch.tensor(np.array(im),dtype=torch.float)
                self.data[i] = get_data.view(-1,512,512)

    def __len__(self):
        return len(self.ann)
        
    def __getitem__(self, index):
        sample = self.data[index]
        sample_ann = self.ann[index]['class']
        
        if self._transform is not None:
            sample = self._transform(sample)
        
        return sample, sample_ann


if __name__ == '__main__':
    data = samho_data("train", in_memory=True)
