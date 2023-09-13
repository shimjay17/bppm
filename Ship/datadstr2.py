import os
import shutil
from dict import get_block_dict
from countmachine import lineno_return
from configloader import configloader
#import random
#random.seed(0) # random성을 고정해주는 함수
#random.

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

line_no = lineno_return()    
jpg_directory = line_no
#['8020','8021','8060','8061','8100','8101','8103','8104','8106','8107','8121','8122','8125','8126'
#,'8127','8128','8129','8132','8134','8138','8140','8141','8148','8149','8150','8151','8159','8160','8164','8165','8173']
block_directory = get_block_dict()  
output_dir =  train_set_path
os.makedirs(output_dir, exist_ok=True)

paths_train, paths_val, paths_test = [], [], []
for folder in block_directory:
    #for sub_name in subblock_unit: 
    #import ipdb; ipdb.set_trace()
    folder_path = os.path.join(snapshot_directory, folder) # -> '/mnt/hdd/jyshim/workspace/Projects/Ship/dataset/img_source/snapshot/image/8020'
    jpg_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]
    # lists  of  all  jpgs paths
    N = len(jpg_files) # number of .jpg files inside the folder
   
   # Divide train/val/test sets , dividing비율 결정
    unit_size = int(N * 0.025) # N//40
    org_train_size = unit_size * 35
    val_size = org_train_size + unit_size
    test_size = val_size + unit_size

    jpg_files_train = jpg_files[:2*unit_size]
    jpg_files_val =jpg_files[org_train_size:val_size]
    jpg_files_test =jpg_files[val_size:test_size]

    paths_train.extend(jpg_files_train)
    paths_val.extend(jpg_files_val)
    paths_test.extend(jpg_files_test)


    for mode in ['train', 'val', 'test']:
     img_paths = eval(f'paths_{mode}')
     for img_path in img_paths:
        src_path = img_path   # .../8020/8020-F51S-SS2-4-6.jpg

        L = src_path.split('/')  # [..., '8020', '8020-F51S-SS2-4-6.jpg']
        line = L[-2] #8020
        root = L[:-2]  # [...]
        img_name = os.path.basename(src_path) # '8020-F51S-SS2-4-6.jpg'

        tar_path =os.path.join(output_dir, f'{mode}_4', line, img_name) # .../others/test_4/8020/8020-F51S-SS2-4-6.jpg
        
        tar_dir = os.path.dirname(tar_path)
        os.makedirs(tar_dir, exist_ok=True)

        print(tar_path) #저장 확인용   
        shutil.copyfile(src_path, tar_path)
       
   









# jpg_files = []
# for folder in jpg_directory:
#     folder_path = os.path.join(directory, folder)
#     jpg_files.extend([os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')])

# num_jpg_dirs = len(jpg_directory)
# #num_files_per_dir = len(jpg_files) // num_jpg_dirs


# jpg_files_reshaped = np.array(jpg_files)[:, None]  #.reshape((num_jpg_dirs, -1))

# ipdb.set_trace()

# for i,folder in enumerate(jpg_directory):
#  unit_size=len(jpg_files[i])//20
#  org_train_size=unit_size*16
#  val_size=org_train_size +unit_size
#  test_size=val_size+unit_size

#  jpg_files_train=jpg_files[i][:2*unit_size]
#  jpg_files_val=jpg_files[i][org_train_size:val_size]
#  jpg_files_test=jpg_files[i][val_size:test_size]

#  train_dir = os.path.join(output_dir, folder+'train_2')
#  test_dir = os.path.join(output_dir, folder+'test_2')
#  val_dir = os.path.join(output_dir, folder+'val_2')
# #paths to store disdtributed data sets->'/mnt/hdd/jyshim/workspace/Datasets/Ship/3d/train'

#  os.makedirs(train_dir, exist_ok=True)
#  os.makedirs(test_dir, exist_ok=True)
#  os.makedirs(val_dir, exist_ok=True)
# #make directions: files generated when not exists/ save directly when the file already exists.

# print("Loading training images...")
# for jpg_file in tqdm(jpg_files_train, total=len(jpg_files_train)):
#     image_path = jpg_file
#     destination_path = os.path.join(train_dir, os.path.relpath(jpg_file, directory))
# #destination_path= /mnt/hdd/jyshim/workspace/Datasets/Ship/3d/train/8020/8020-B51P-BS1-1-3.jpg 
#     os.makedirs(os.path.dirname(destination_path), exist_ok=True)

#     ipdb.set_trace()

#     shutil.copyfile(image_path, destination_path)
# #relpath: relative path btw jpg_file & directory
# #shutil: offers a number of high-level operations on files and collections of files->shutil.copyfile(from_path_file, to_path_file & file_name)
# print("Loading test images...")
# for jpg_file in tqdm(jpg_files_test, total=len(jpg_files_test)):
#     image_path = jpg_file
#     destination_path = os.path.join(test_dir, os.path.relpath(jpg_file, directory))
#     os.makedirs(os.path.dirname(destination_path), exist_ok=True)
#     shutil.copyfile(image_path, destination_path)

# print("Loading validation images...")
# for jpg_file in tqdm(jpg_files_val, total=len(jpg_files_val)):
#     image_path = jpg_file
#     destination_path = os.path.join(val_dir, os.path.relpath(jpg_file, directory))
#     os.makedirs(os.path.dirname(destination_path), exist_ok=True)
#     shutil.copyfile(image_path, destination_path)  

