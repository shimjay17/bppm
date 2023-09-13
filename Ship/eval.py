#test inference 코드
import os
import torch
import torch.optim as optim
import torch.nn as nn
import sys
from dataset import ship_dataset
from dataset_test import test_dataset
from torch.utils.data import DataLoader
from transform import test_transform
from test import test
from utils import load_checkpoint, ensemble
from torchvision.models import efficientnet_v2_s
from countmachine import class_num, unit_combno_return
from configloader import configloader
from step_detection.step_det import step_det

# from Ship.dataset import ship_dataset
# from Ship.dataset_test import test_dataset
# from torch.utils.data import DataLoader
# from Ship.transform import test_transform
# from Ship.test import test
# from Ship.utils import load_checkpoint, ensemble
# from torchvision.models import efficientnet_v2_s
# from Ship.countmachine import class_num, unit_combno_return
# from Ship.configloader import configloader
# from Ship.step_detection.step_det import step_det

import fire
import ipdb

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

def eval(result_path):

    
    main_no = class_num()
    unit_no = unit_combno_return() 
    
    

    print('class number:', main_no)
    print('unit number:', unit_no)
    # x = int(input())

    model_name = 'efficientnetv2_s'

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    # if x == 0:
    #     model = CNN().to(device)  
    #     print('Chosen model: CNN')
    # elif x == 1:
    #     model = ResNet(block, [3, 4, 23, 3], 3, 186).to(device)
    #     print('Chosen model: ResNet')
    # elif x == 2:
    model = efficientnet_v2_s(num_classes=main_no, num_classes2=unit_no).to(device)

    #print number of parameters in model
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    # print('Chosen model: efficientnet_v2_s')

    # else:
    #     raise ValueError('Model number not recognized...')

    model = nn.DataParallel(model, device_ids=[0])

    # val_dataset = ship_dataset(block_dict_path,unit_dict_path, mothership_dict_path, os.path.join(test_set_path, 'val'), transform=test_transform)
    # test1_dataset = ship_dataset(block_dict_path,unit_dict_path, mothership_dict_path, os.path.join(test_set_path, 'test1'), transform=test_transform)
    # test2_dataset = ship_dataset(block_dict_path,unit_dict_path, mothership_dict_path, os.path.join(test_set_path, 'test2'), transform=test_transform)
    # test3_dataset = ship_dataset(block_dict_path,unit_dict_path, mothership_dict_path, os.path.join(test_set_path, 'test3'), transform=test_transform)
    # test_dataset = ship_dataset(block_dict_path,unit_dict_path, mothership_dict_path, test_set_path, transform=test_transform)
    testf_dataset = test_dataset(block_dict_path,unit_dict_path, mothership_dict_path, result_path, transform=test_transform)

    # val_dataset = ship_dataset(block_dict_path, os.path.join(dataset_path, 'val'), transform=test_transform)

    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    # test1_dataloader = DataLoader(dataset=test1_dataset, batch_size=batch_size, shuffle=False)
    # test2_dataloader = DataLoader(dataset=test2_dataset, batch_size=batch_size, shuffle=False)
    # test3_dataloader = DataLoader(dataset=test3_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=testf_dataset, batch_size=batch_size, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    #weights = torch.tensor(class_weights).to(device) 

    if class_weights is not None:
        weights = torch.tensor(class_weights).to(device)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Initialize network

    load_checkpoint(model, None, checkpoint_path)
    # test(model, val_dataloader, optimizer, criterion, device, relationship_dict_path, desc = "val")
    # test(model, test1_dataloader, optimizer, criterion, device, relationship_dict_path, desc = "test1")
    # test(model, test2_dataloader, optimizer, criterion, device, relationship_dict_path, desc = "test2")
    # test(model, test3_dataloader, optimizer, criterion, device, relationship_dict_path, desc = "test3")
    test(model, test_dataloader, optimizer, criterion, device, relationship_dict_path, result_path, desc = "test")


if __name__ == "__main__":

    
    # # ipdb.set_trace()
    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_directory>")
        exit(1)
    dir_name = sys.argv[1]
    del sys.argv[1]
    
    try:
        eval(dir_name)
    except KeyboardInterrupt:
        print('Quit the program...')
    exit(0)


    # eval('/home/admin/workspace/bppm/output/test2_2023-09-08_16-45-54/stitching/final_results')


# if __name__ == "__main__":
#     fire.Fire()



    

