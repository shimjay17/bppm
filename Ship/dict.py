import os
from glob import glob
import natsort
import json
from configloader import configloader

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
os.makedirs(json_directory, exist_ok=True)


def load_mothership_dict(mothership_dict_path):
    with open(mothership_dict_path, 'r') as f:
        mothership = json.load(f)
    return mothership

def extract_number_block_names(image_path, mothership): #블록네임 빌드를 위해 하이픈 '-'단위로 basename 분할
    #image_name = os.path.basename(image_path)
    s =os.path.dirname(image_path).split('/')[-1]
    #s = image_name.split('-') # 8020,  B51P0  ,  1, 1 
    mothership_value = None
    for  key, value in mothership.items() :
            if key == s:
                mothership_value = value

    #number_name = s[0] # 8020
    s_2 = os.path.basename(image_path).split('-')
    rvsd_line = mothership_value
    block_name = s_2[1] # B51P0
    return rvsd_line, block_name         #, number_name 

def extract_unit_block_names(image_path, mothership):

    s =os.path.dirname(image_path).split('/')[-1]
    #s = image_name.split('-') # 8020,  B51P0  ,  1, 1 
    
    mothership_value = None
    for key, value in mothership.items():
            if key == s:
                mothership_value = value

    image_name = os.path.basename(image_path)
    s = image_name.split('-') # 8020,  B51P0  ,  1, 1 
    # if len(s[:-2]) == 2:
    #     return None
    rvsd_line = mothership_value
    psd_unitb_names = '-'.join(s[1:-2])
    unitb_names = '-'.join([rvsd_line, psd_unitb_names])
    return unitb_names
 
def call_main_block(train_set_path, mothership):
    modes = ['train', 'val','test1','test2']
    images_list = [] # 나중에 쓸 array 정의
    for mode in modes:
        images_list.extend(glob(f'{train_set_path}/{mode}/*/*.jpg'))

    number_block_pairs = [extract_number_block_names(image_path, mothership) for image_path in images_list] # image_list = [jpgs in train, val, test]
    number_block_pairs_set = [f'{number_name}-{block_name}' for number_name, block_name in number_block_pairs if number_name and block_name] # 8020-B51P0
    number_block_pairs_set = natsort.natsorted(list(set(number_block_pairs_set))) # window Linux정렬방식이 달라서 ///둘중 하나로 변환: 윈도우는 1이 첫글자인 것부터///리눅스는 integer 순
    number_block_pairs_dict = {number_block_pair: i for i, number_block_pair in enumerate(number_block_pairs_set)} # 8020_B51P0 :0

    return number_block_pairs_dict #list of main block numbering

def call_unit_block(train_set_path, mothership):
    unit_block_list = []
    modes = ['train', 'val','test1','test2']
    images_list = [] # 나중에 쓸 array 정의
    for mode in modes:
        images_list.extend(glob(f'{train_set_path}/{mode}/*/*.jpg'))
    
    
    unit_block_list =[extract_unit_block_names(image_path, mothership) for image_path in images_list]
    unit_block_list = list(filter(None, unit_block_list))  # Filter out None values
    unit_block_list = natsort.natsorted(list(set(unit_block_list))) # window Linux정렬방식이 달라서 ///둘중 하나로 변환: 윈도우는 1이 첫글자인 것부터///리눅스는 integer 순
    unit_block_list = {number_block_pair: i for i, number_block_pair in enumerate(unit_block_list)}
    print(unit_block_list)
    return unit_block_list

def get_block_dict():   #메인블록의 json파일 생성 및 저장
    mothership = load_mothership_dict(mothership_dict_path)
    main_block_dict = call_main_block(train_set_path, mothership)
    
    with open(block_dict_path, 'w') as f:
        json.dump(main_block_dict, f, indent=4)
        print(f'Main block dictionary saved to: {block_dict_path}')

    return main_block_dict

def get_combined_block_dict():  #'메인블록+단위블록'의 json파일 생성 및 저장
    mothership = load_mothership_dict(mothership_dict_path)
    unit_block_dict = call_unit_block(train_set_path, mothership)
    
    
    with open(unit_dict_path, 'w') as f:
        json.dump(unit_block_dict, f, indent=4)
        print(f'Unit block dictionary saved to: {unit_dict_path}')
    return unit_block_dict

def main():
    get_block_dict()
    get_combined_block_dict()
if __name__ == '__main__':
    main()








