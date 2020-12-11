#!/bin/bash

PROJECT_PATH="/workspace/home/projects/VS_Seg_defaced"

RESUlTS_FOLDER_NAME_T1="UNet2d5_Att_Hard_T1"
RESUlTS_FOLDER_NAME_T2="UNet2d5_Att_Hard_T2"

cd $PROJECT_PATH


python3 VS_train.py     --results_folder_name $RESUlTS_FOLDER_NAME_T1 --dataset T1   2> train_error_log3_T1.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME_T2 --dataset T2   2> inference_error_log3_T2_fixed_segs.txt VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME3_T2 --dataset T2                              2> inference_error_log3_T2_fixed_segs.txt