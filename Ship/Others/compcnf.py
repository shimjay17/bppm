import os
import ipdb
import shutil
from dict import get_block_dict

directory_1 = '/mnt/hdd/jyshim/workspace/Datasets/Ship/3d/others/20230526/image'
directory_2 = '/mnt/hdd/jyshim/workspace/Datasets/Ship/3d/others/20230526/snapshot/image'
#dir_out='/mnt/hdd/jyshim/workspace/Datasets/Ship/3d/others'

jpg_directory = ['8060','8061','8104','8106','8107','8121','8126','8127','8134','8138','8141','8148','8149']

for linenum in jpg_directory:
 jpg_dir_1 = os.path.join(directory_1,linenum) 
 
# jpg_dir_1 = []

# for path in jpg_files_1:
#  filename, ext = os.path.splitext(path) 
#  name_split = os.path.basename(filename).split('_')[0].split('-')
#  mod_name =  '-'.join(name_split[:3]) #+ ext
#  jpg_dir_1.append(mod_name)

 jpg_files_1 = [os.path.join(jpg_dir_1, filename) for filename in os.listdir(jpg_dir_1) if filename.endswith('.jpg')]

 jpg_path_1 = ['-'.join(os.path.splitext(os.path.basename(filename))[0].split('-')[:3]) for filename in os.listdir(jpg_dir_1) if filename.endswith('.jpg')]
 



 parts_jpg_dir_1 = [os.path.basename(path).split('-') for path in jpg_files_1]
#jpg_dir_1 = ['-'.join(path[:3]) for path in parts_jpg_dir_1]

#print(jpg_dir_1)
 jpg_dir_2 = os.path.join(directory_2,linenum) 
    
 jpg_files_2 = [os.path.join(jpg_dir_2, filename) for filename in os.listdir(jpg_dir_2) if filename.endswith('.jpg')]
#parts_jpg_dir_2 = [os.path.basename(path).split('_')[0].split('-') for path in jpg_files_2]
#jpg_dir_2 = ['-'.join(path[:3]) for path in parts_jpg_dir_2]
#print(jpg_dir_2)
 jpg_path_2 = ['-'.join(os.path.splitext(os.path.basename(filename))[0].split('-')[:3]) for filename in os.listdir(jpg_dir_2) if filename.endswith('.jpg')]

 print(jpg_path_2)
 #ipdb.set_trace()
 for jpg_path ,jpg_path_full in zip(jpg_files_1, jpg_path_1):

  if jpg_path_full not in jpg_path_2 :
    os.remove(jpg_path)
    #jpg_ext_wo=
    print('success')
  
  
# for path in paths_2_comm:
#     # Find the corresponding JPEG file in directory_1
#     for jpg_path in jpg_files_1:
#         basename = os.path.splitext(os.path.basename(jpg_path))[0]
#         if basename.startswith(path):
#             output_directory = os.path.join(dir_out, basename)
#             os.makedirs(output_directory, exist_ok=True)
#             output_file = os.path.join(output_directory, basename + '.jpg')
#             shutil.copyfile(jpg_path, output_file)
#             break
# #ipdb.set_trace()

# for path in paths_2_comm:
#     # Find the corresponding JPEG file in directory_2
#     for jpg_path in jpg_files_2:
#         basename = os.path.splitext(os.path.basename(jpg_path))[0]
#         if basename.startswith(path):
#             output_directory = os.path.join(dir_out, basename)
#             os.makedirs(output_directory, exist_ok=True)
#             output_file = os.path.join(output_directory, basename + '.jpg')
#             shutil.copyfile(jpg_path, output_file)
#             break
                      