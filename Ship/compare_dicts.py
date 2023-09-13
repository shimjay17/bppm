from configloader import configloader
import json
from collections import defaultdict

if __name__ == '__main__':
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

    with open(block_dict_path, 'r') as f:
        block_dict = json.load(f)

    with open(unit_dict_path, 'r') as f:
        unit_dict = json.load(f)

    block_relationship = defaultdict(list)  # Change here
    for key1, value1 in block_dict.items():
        for key2, value2 in unit_dict.items():  
            if key1 in key2:
                block_relationship[value1].append(value2)  # Change here

    with open(relationship_dict_path, 'w') as f:
        json.dump(block_relationship, f, indent=4)
