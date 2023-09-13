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
# from models import resnet
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from torchsampler import ImbalancedDatasetSampler

# Xavier Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Early Stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_accuracy_test(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    loss_valid = 0
    criterion = nn.CrossEntropyLoss()
    targets_list = []
    predicted_labels_list = []
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        # logits1, probas1 = model(features[:,:,:56,:56]) # top left
        # logits2, probas2 = model(features[:,:,8:,:56]) # top right
        # logits3, probas3 = model(features[:,:,:56,8:]) # bot left
        # logits4, probas4 = model(features[:,:,8:,8:]) # bot right
        # logits5, probas5 = model(features[:,:,4:60,4:60]) # center

        # probas = (probas1+probas2+probas3+probas4+probas5)/5
        # logits = (logits1+logits2+logits3+logits4+logits5)/5
        
        # _, predicted_labels = torch.max(probas, 1)

        # outputs1 = model(features[:,:,:224,:224]) # top left
        # outputs2 = model(features[:,:,8:,:224]) # top right
        # outputs3 = model(features[:,:,:56,8:]) # bot left
        # outputs4 = model(features[:,:,8:,8:]) # bot right
        # outputs5 = model(features[:,:,4:60,4:60]) # center

        # outputs = (outputs1+outputs2+outputs3+outputs4+outputs5)/5
        outputs = model(features)
        _, predicted_labels = torch.max(outputs, 1)

        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        targets_list.append(targets.cpu().numpy())
        predicted_labels_list.append(predicted_labels.cpu().numpy())
        # loss_valid += F.cross_entropy(logits, targets)

    for idx in range(len(predicted_labels_list)):
        if idx == 0:
            predicted_labels_list_final = predicted_labels_list[idx]
            targets_list_final = targets_list[idx]
        else:
            predicted_labels_list_final = np.concatenate((predicted_labels_list_final, predicted_labels_list[idx]), axis=None)
            targets_list_final = np.concatenate((targets_list_final, targets_list[idx]), axis=None)
    return correct_pred.float()/num_examples * 100, loss_valid, predicted_labels_list_final, targets_list_final

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(features)
        _, predicted_labels = torch.max(outputs, 1)

        # logits, probas = model(features)
        # _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def step_det():
    # Set random seed for reproducibility
    manualSeed = 0
    print("Random Seed:", manualSeed)

    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configuration
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 40
    l2_weight = 3e-4
    batch_size = 8

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    # Set up dataset
    custom_transform_train = transforms.Compose([
        # transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])

    custom_transform_test = transforms.Compose([
        # transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])

    train_dataset = samho_data("train", transform=custom_transform_train, in_memory=True)
    test_dataset = samho_data("test", transform=custom_transform_test, in_memory=True)

    # train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Load dataset: Done")

    # Set up model
    # model = resnet.ResNet18()
    model = models.resnet18(pretrained=True)   #load resnet18 model
    num_features = model.fc.in_features     #extract fc layers features
    model.fc = nn.Linear(num_features, 3)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight)

    class_weights = [15.66666667, 1.18238994, 0.4783715]
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')  #(set loss function)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
    model.apply(init_weights)

    early_stopper = EarlyStopper(patience=3, min_delta=0)

    # Traing
    start_time = time.time()
    min_val = 0
    best_test_acc = 0

    print("Training!!!")
    for epoch in range(num_epochs):
        print('Epoch: %03d/%03d'% (epoch+1, num_epochs))
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)
            
            # Forward and backpropagation
            # logits, probas = model(features, dropout=False)
            # loss = F.cross_entropy(logits, targets)
            # running_loss = loss.item()

            outputs = model(features)
            _, logits = torch.max(outputs, 1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            
            loss.backward()
            
            # Update model params
            optimizer.step()

        lr_scheduler.step()
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device)
            test_acc, loss_test, outputs_test, targets_test = compute_accuracy_test(model, test_loader, device)

            print('Epoch: %03d/%03d | Train: %.3f%% | Test: %.3f%%' % (
                epoch+1, num_epochs, 
                train_acc,
                test_acc))
            # target_names = ['class 0', 'class 1', 'class 2']
            # print(classification_report(targets_test, outputs_test, target_names = target_names))
            # print('Confusion matrix:')
            # print(confusion_matrix(targets_test, outputs_test))

            if test_acc > best_test_acc:
                best_epoch = epoch + 1
                best_test_acc = test_acc
                best_pred_test = outputs_test
                best_targets_test = targets_test

        torch.save(model.state_dict(), './checkpoint/epoch_{}.pth'.format(epoch + 1))
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
        # Early Stopping
        if early_stopper.early_stop(loss_test):        
            print("Early stopped at epoch {}.".format(epoch + 1))
            break

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    print('Best epoch: %03d' % (epoch+1))
    print('Result best predictions:', best_pred_test)
    print('Result targets:', best_targets_test)
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(best_targets_test, best_pred_test, target_names = target_names))
    print('Confusion matrix:')
    print(confusion_matrix(best_targets_test, best_pred_test))

    # cm = confusion_matrix(best_targets_test, best_pred_test)
    # per_class_accuracies = []

    # # Calculate the accuracy for each one of our classes
    # for idx in range(0, 3):
    #     # True negatives are all the samples that are not our current GT class (not the current row) 
    #     # and were not predicted as the current class (not the current column)
    #     true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        
    #     # True positives are all the samples of our current GT class that were predicted as such
    #     true_positives = cm[idx, idx]
        
    #     # The accuracy for the current class is the ratio between correct predictions to all predictions
    #     per_class_accuracies.append((true_positives + true_negatives) / np.sum(cm))
    # print('Per class accuracies: ', per_class_accuracies)


if __name__ == '__main__':
    step_det()