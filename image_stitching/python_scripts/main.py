import cv2
import os
import sys
import glob
import shutil
import yaml
from easydict import EasyDict as edict
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from stitch.stitcher import ImageStitcherNode
from utils import get_logger, unsharp_masking, truncate_image_vertical

import ipdb

def main(dir_name):
    ########### Argument Parsing ###########
    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--config_path', type=str, default='./config/stitching.yaml')
    args = parser.parse_args()
    ########################################
    
    with open(args.config_path, 'r') as f:
        conf = edict(yaml.load(f, Loader=yaml.SafeLoader))
    for k, v in vars(args).items():
        conf[k] = v
    
    input_dir = os.path.join(conf.data.image_dir, dir_name)
        
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        exit(1)

    # Set logger
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(conf.data.output_dir, dir_name+'_'+log_name, 'stitching')
    print(f'Log saved to: {log_dir}')

    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, 'src_images'))
    os.makedirs(os.path.join(log_dir, 'out_images'))
    os.makedirs(os.path.join(log_dir, 'final_results'))
    if conf.algorithm.view_for_user.make_view:
        os.makedirs(os.path.join(log_dir, 'views'))
        if not conf.algorithm.view_for_user.stitch in ['all', 'hori', 'vert']:
            print("conf.algorithm.view_for_user.stitch not ['all', 'hori', 'vert']")
            exit(1)

    logger = None #get_logger(os.path.join(log_dir, 'log.txt'))
    conf.log_dir = log_dir
    conf.logger = None #logger

    # Extract upper and lower pixels to remove
    pxl_cut_upper, pxl_cut_lower = map(int, conf.data.pixel_cuts.split(',')) # just for print
    print(f'Upper {pxl_cut_upper} lines to be removed')
    print(f'Lower {pxl_cut_lower} lines to be removed')
    
    # Prepare input images
    src_images = []
    img_ext = ["*.jpg", "*.jpeg", "*.png"]
    for ext in img_ext:
        src_images.extend(glob.glob(os.path.join(input_dir, ext))) #get all image paths in the input folder
    image_groups = dict()
    for image_path in src_images:
        image_name = os.path.basename(image_path)
        group_id, cam_id, tilt, _ , _ = image_name.split('_') #split to get the info
        save_dir = os.path.join(conf.log_dir, 'src_images', group_id, cam_id, tilt) #make directory to save the preprocessed images
        os.makedirs(save_dir, exist_ok=True) 

        save_path = os.path.join(save_dir, image_name)
        
        img = cv2.imread(image_path) # read image from the path
        H, _, _ = img.shape 
        img = truncate_image_vertical(img, pxl_cut_upper, pxl_cut_lower) # cut the date info at the top
        cv2.imwrite(save_path, img) #save image

        if group_id not in image_groups.keys(): # update dictionary of groups. key=group, value=list of image paths
            image_groups[group_id] = []
        image_groups[group_id].append(save_path)
        
    conf.image_groups = image_groups 

    num_images_per_group = {group_id: len(group_images) for group_id, group_images in image_groups.items()} # count how many images per group for print
    print(f'Number of images: {num_images_per_group}')
    sys.stdout.flush()
    
    node = ImageStitcherNode(conf) 
    node.stitch_all()     

    # erase saved mid-process images 
    if not conf.data.save_process:
        shutil.rmtree(os.path.join(log_dir, 'src_images'))
        shutil.rmtree(os.path.join(log_dir, 'out_images'))

if __name__ == '__main__':
    # input directory
    if len(sys.argv) < 2:
        print("Usage: python inference.py <input_directory>")
        exit(1)
    dir_name = sys.argv[1]
    del sys.argv[1]
    
    try:
        main(dir_name)
    except KeyboardInterrupt:
        print('Quit the program...')
    exit(0)
