#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <cuda_visible_device> <config_path>"
    exit 1
fi

# Assign input arguments to variables
cuda_visible_device=$1
config_path=$2

# Export the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$cuda_visible_device

# Run the Python script with the provided config path
python main.py $config_path
