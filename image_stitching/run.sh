#!/bin/bash

# check if argument is provided 
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input_dir"
    exit 1
fi

input_dir=$1

python ./python_scripts/main.py $input_dir
