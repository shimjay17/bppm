import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from configloader import configloader
#############################################################################################################################################
# print('select from jpg_counter / class_num / lineno_return / unit_combno_return:')
# x = input()

# # 호선 리스트, jpg 갯수 등 반환하고자 하는 정보에 따라 jpg 또는 json file 의 path 입력

# if x == 'jpg_counter':
#     jpg_directory = input("Enter the JPG file directory: ") # 전체 jpg 갯수 카운트
# elif x == 'class_num':
#     clss_jsn_directory = input("Enter the 'main_block_dict' JSON file directory: ") #메인블록 (e.g. 8020-BP1)의 갯수 카운트
# elif x == 'lineno_return':
#     lineno_directory = input("Enter the snapshot/image file directory: ") #호선의 리스트 반환

# elif x == 'unit_combno_return' :
#     unit_combno_return_directory = input("Enter the 'unit_block_dict' JSON file directory: ") #메인블록+단위블록(e.g. 8020-BP1-SS1)의 갯수 반환
# else:
#     print("Invalid option. Please try again.")
#     sys.exit(1)
#############################################유저 입력 버젼(#해제 후 사용)###########################################################################################

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

def jpg_counter():
    # Initialize a counter for JPG files
    num_jpg_files = 0
    #directory = os.path.join(main_directory, jpg_directory)

    # Recursively iterate over the directory and its subdirectories
    for root, dirs, files in os.walk(jpg_directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if the file is a JPEG file
            if file.lower().endswith('.jpg'):
                num_jpg_files +=1

    print(f"Number of JPG files in the directory: {num_jpg_files}")
    return num_jpg_files


def class_num(): 
     #directory = os.path.join(main_directory, clss_jsn_directory)
 # Load the JSON file
    with open(block_dict_path) as f:
        data = json.load(f)

 # Access the last label
    no_elements = len(data)

    # print(f"Number of mainblocks: {no_elements}")
    
    return no_elements
   

def lineno_return():
    #directory=os.path.join(main_directory,lineno_directory)
    line_no=os.listdir(snapshot_directory)
    line_no.sort()
    #os.remove('list.txt')
    print(f"Lists of lines in the directory: {line_no}")
    return line_no


def unit_combno_return():
    with open(unit_dict_path) as f:
        data = json.load(f)
   
    unit_combno = len(data)
    print(f"Number of unitblocks: {unit_combno}")
    
    return unit_combno
###############################################################################
# def main():

#  if x == 'jpg_counter':
#   jpg_counter()

#  elif x == 'class_num':
#   class_num()

#  elif x == 'lineno_return':
#   lineno_return()
 


#  elif x == 'unit_combno_return':
#   unit_combno_return()
###################################유저 입력 버젼(#해제 후 사용)###################

def main():

  #jpg_counter()

  class_num() #main block

  #lineno_return() 
 
  unit_combno_return() # unit block

if __name__ == '__main__':
    
    main()


