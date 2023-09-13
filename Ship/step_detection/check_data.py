import json
import numpy as np
import sklearn

def check_data():
    # Train data
    train_data_path = './data/train.json'
    train_f = open(train_data_path)
    train_ann = json.load(train_f)
    train_labels = np.zeros(3)
    train_list_labels = []

    for i in range(len(train_ann)):
        train_labels[train_ann[i]['class']] += 1
        train_list_labels.append(train_ann[i]['class'])

    # Test data
    test_data_path = './data/test.json'
    test_f = open(test_data_path)
    test_ann = json.load(test_f)
    test_labels = np.zeros(3)

    for i in range(len(test_ann)):
        test_labels[test_ann[i]['class']] += 1

    print('Train data labels: ', train_labels)
    print('Test data labels: ', test_labels)

    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(train_list_labels), y=np.array(train_list_labels))
    print('Class weights: ', class_weights)

if __name__ == '__main__':
    check_data()