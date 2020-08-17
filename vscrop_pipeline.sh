#!/bin/bash

pip install git+https://github.com/Project-MONAI/MONAI.git@0d197e6bea9dd2244c63b80a80b464ef23a5aab9
pip install nibabel
pip install natsort

PROJECT_PATH="/workspace/home/projects/VS_Seg"
RESUlTS_FOLDER_NAME="UNet_testing"

cd $PROJECT_PATH

python3 VS_train.py --train_batch_size 2 --results_folder_name $RESUlTS_FOLDER_NAME 2> train_error_log.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME 2> inference_error_log.txt