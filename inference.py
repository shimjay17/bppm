import os
import sys
import argparse
import yaml
from easydict import EasyDict as edict
import subprocess

import ipdb

#input directory
if len(sys.argv) < 2:
    print("Usage: python inference.py <input_directory>")
    sys.exit(1)
dir_name = sys.argv[1]
del sys.argv[1]

# load stitching config for input and output directory info
parser = argparse.ArgumentParser(description='Image Stitching')
parser.add_argument('--config_path', type=str, default='./config/stitching.yaml')
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    conf = edict(yaml.load(f, Loader=yaml.SafeLoader))
for k, v in vars(args).items():
    conf[k] = v

input_dir = os.path.join(conf.data.image_dir, dir_name)
    
if not os.path.exists(input_dir):
    print(f"Error: Directory '{input_dir}' does not exist.")
    sys.exit(1)

#start stitching
print("Start stitching...")
cmd = ["python", "./image_stitching/python_scripts/main.py", dir_name]
try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        result_path = line
    process.communicate()
    result_path = result_path.split()[-1]
   
except Exception as e:
    print(f"Error executing command: {e}")
    print(f"Command output:{e.output}")
    print(f"Command error output: {e.stderr}")

cmd1 = ["python", "Ship/eval.py", result_path]

try:
    subprocess.run(cmd1)
except Exception as e:
    print(f"Error executing command: {e}")