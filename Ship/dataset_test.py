#이미지/라벨 로더

import os
import glob
import torch
import torchvision

from torch.utils.data import Dataset
import json
from torchvision.transforms import ToPILImage, Resize
import torch
from torch.utils.data import Dataset, DataLoader
import re

class test_dataset(Dataset):
    def __init__(self, block_dict_path, unit_dict_path, mothership_dict_path, dataset_path, transform=None, target_transform=None, num_classes=None, unit_combno=None):
        self.num_classes = num_classes
        self.unit_combno = unit_combno
        print('Loading main block dictionary data...')
        with open(block_dict_path, 'r') as file:
            main_block_json_data = json.load(file)
        
        print('Loading unit block dictionary data...')
        with open(unit_dict_path, 'r') as file:
            unit_json_data = json.load(file)

        print('Loading mothership dictionary data...')
        with open(mothership_dict_path, 'r') as file:
            mothership_json_data = json.load(file)

        self.path = dataset_path

        jpg_paths = glob.glob(os.path.join(self.path, '*.jpg'))
        png_paths = glob.glob(os.path.join(self.path, '*.png'))
        jpeg_paths = glob.glob(os.path.join(self.path, '*.jpeg'))

        self.class_list = main_block_json_data
        self.unit_class_list = unit_json_data
        self.mothership = mothership_json_data
        self.transform = transform
        self.to_pil = ToPILImage()
        self.image_paths = sorted(jpg_paths + png_paths + jpeg_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):

        img_paths = self.image_paths[i]
        image = self.loadImage(img_paths)
        image = torchvision.transforms.Resize((512,512), antialias=True)(image)
        gt_name = self.load_gt(img_paths)
        camid = self.load_cam(img_paths)

        if self.transform:
            image = self.to_pil(image)
            image = self.transform(image)

        return image, gt_name, camid

    def loadImage(self, path):
        img = torchvision.io.read_image(path).type(torch.float32)[:3, :, :]
        img = img / 255.0
        img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img
    
    def load_gt(self, path):
        gt = path.split('/')[-1]

        return gt
    
    def load_cam(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        camid = name.split('_')[1]
        return camid


