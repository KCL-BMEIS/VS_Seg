#!/bin/bash

PROJECT_PATH="/workspace/home/projects/VS_Seg"

RESUlTS_FOLDER_NAME1="UNet2d5_noAtt_noHard"
RESUlTS_FOLDER_NAME2="UNet2d5_Att_noHard"
RESUlTS_FOLDER_NAME3="UNet2d5_Att_Hard"

cd $PROJECT_PATH

python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME1 --no_attention --no_hardness 2> train_error_log1.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME2                --no_hardness 2> train_error_log2.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME3                              2> train_error_log3.txt

python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME1 --no_attention --no_hardness 2> inference_error_log1.txt &
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME2                --no_hardness 2> inference_error_log2.txt &
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME3                              2> inference_error_log3.txt