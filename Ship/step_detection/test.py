import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms, models

from step_detection.datasets import samho_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def step_test(input, device):
    manualSeed = 0
    print("Random Seed:", manualSeed)

    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 8


    # Set up dataset
    # custom_transform_test = transforms.Compose([
    #     # transforms.Resize((56, 56)),
    #     transforms.Resize((224, 224)),
    #     # transforms.ToTensor(),
    #     transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    # ])

    # test_dataset = samho_data("test", transform=custom_transform_test, in_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("Load dataset: Done")

    # Set up model
    model = models.resnet18(pretrained=False)   #load resnet18 model
    num_features = model.fc.in_features     #extract fc layers features
    model.fc = nn.Linear(num_features, 3)
    model = model.to(device)
    # breakpoint()
    
    weights = torch.load('/home/admin/workspace/bppm/Ship/step_detection/checkpoint/epoch_11.pth', map_location=device)
    model.load_state_dict(weights)

    criterion = nn.CrossEntropyLoss()

    # Testing
    start_time = time.time()  
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.set_grad_enabled(False):
        outputs = model(input)
        probability, predicted_labels = torch.max(outputs, 1)
        # for i, (features, targets) in enumerate(test_loader):
            # features = features.to(device)
            # targets = targets.to(device)

            # num_examples += targets.size(0)
            # correct_pred += (predicted_labels == targets).sum()

    # test_acc = correct_pred.float()/num_examples * 100
    # print(f'Test acc: {test_acc}')

    return predicted_labels, probability

        
if __name__ == '__main__':
    step_test()
