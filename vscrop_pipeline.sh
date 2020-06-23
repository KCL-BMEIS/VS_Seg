#!/bin/bash

pip install git+https://github.com/Project-MONAI/MONAI.git@eb104391d18e7c8325d9e3efd5b56f7c2f7fad4a
pip install natsort

PROJECT_PATH="/workspace/home/projects/VS_Segmentation"
MODEL_PATH="/workspace/home/projects/VS_Segmentation/best_metric_model.pth"
RESUlTS_FOLDER_NAME="BatchSize4"

cd $PROJECT_PATH

python3 VS_train.py --train_batch_size 4 --results_folder_name $RESUlTS_FOLDER_NAME 2> train_error_log.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME 2> inference_error_log.txt