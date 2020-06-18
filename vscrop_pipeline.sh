#!/bin/bash

pip install git+https://github.com/Project-MONAI/MONAI.git@eb104391d18e7c8325d9e3efd5b56f7c2f7fad4a

PROJECT_PATH="/workspace/home/projects/VS_Segmentation"
MODEL_PATH="/workspace/home/projects/VS_Segmentation/best_metric_model.pth"

cd $PROJECT_PATH

python3 VS_train.py
python3 VS_inference.py


