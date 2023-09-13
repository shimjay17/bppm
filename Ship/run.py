#학습을 진행하는 파일

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ship_dataset

from transform import train_transform, test_transform
from train import train
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from countmachine import class_num, unit_combno_return
from configloader import configloader
from step_detection.step_det import step_det

if __name__ == "__main__":

    (learning_rate,
        num_epochs,
        batch_size,
        weight_decay,
        class_weights,
        train_set_path,
        test_set_path,
        block_dict_path,
        unit_dict_path,
        relationship_dict_path,
        mothership_dict_path,
        results_dict_path,
        checkpoint_path,
        snapshot_directory,
        json_directory,
        jpg_directory,
        results_directory) = configloader()

    # x = int(input())

    # x = 2

    model_name = "efficientnetv2_s"

    main_no = class_num()
    unit_no = unit_combno_return() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = efficientnet_v2_s(num_classes=main_no, num_classes2=unit_no).to(device)
    model = nn.DataParallel(model, device_ids=[0])

    train_dataset = ship_dataset(block_dict_path, unit_dict_path, mothership_dict_path, os.path.join(train_set_path, 'train'), transform=train_transform)
    validation_dataset = ship_dataset(block_dict_path, unit_dict_path, mothership_dict_path, os.path.join(train_set_path, 'val'), transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 


    for param in model.parameters():
        param.requires_grad = False

    #weights = torch.tensor(class_weights).to(device) 

    if class_weights is not None:
        weights = torch.tensor(class_weights).to(device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs, model_name) # type: ignore

