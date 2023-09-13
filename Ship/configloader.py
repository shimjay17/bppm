import os
import yaml

def configloader():
    file_path = os.path.abspath(__file__) # /home/admin/workspace/bppm/Ship/configloader.py
    workspace_folder = os.path.dirname(os.path.dirname(file_path)) #/home/admin/workspace/bppm/
    config_loc = 'Ship/config.yaml'
    # config = '/mnt/hdd/jyshim/workspace/Projects/Ship/config.yaml'

    config_path = os.path.join(workspace_folder, config_loc)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    class_weights = config['class_weights']
    train_set_path = os.path.join(workspace_folder, config['train_set_path'])
    test_set_path = os.path.join(workspace_folder, config['test_set_path'])
    block_dict_path = os.path.join(workspace_folder, config['main_dict_path'])
    unit_dict_path = os.path.join(workspace_folder, config['unit_dict_path'])
    relationship_dict_path = os.path.join(workspace_folder, config['relationship_dict_path'])
    mothership_dict_path = os.path.join(workspace_folder, config['mothership_dict_path'])
    result_dict_path = os.path.join(workspace_folder, config['result_dict_path'])
    checkpoint_path = os.path.join(workspace_folder, config['checkpoint_path'])
    snapshot_directory = os.path.join(workspace_folder, config['snapshot_directory'])
    json_directory = os.path.join(workspace_folder, config['json_directory'])
    jpg_directory = os.path.join(workspace_folder, config['jpg_directory'])
    results_directory = os.path.join(workspace_folder, config['results_directory'])
    return learning_rate, num_epochs, batch_size, weight_decay, class_weights, train_set_path, test_set_path, block_dict_path, unit_dict_path,relationship_dict_path, mothership_dict_path, result_dict_path, checkpoint_path,json_directory, snapshot_directory,jpg_directory,results_directory