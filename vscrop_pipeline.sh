#!/bin/bash

PROJECT_PATH="/workspace/home/projects/VS_Seg"

RESUlTS_FOLDER_NAME1_T1="UNet2d5_noAtt_noHard_T1"
RESUlTS_FOLDER_NAME2_T1="UNet2d5_Att_noHard_T1"
RESUlTS_FOLDER_NAME3_T1="UNet2d5_Att_Hard_T1"

RESUlTS_FOLDER_NAME1_T2="UNet2d5_noAtt_noHard_T2"
RESUlTS_FOLDER_NAME2_T2="UNet2d5_Att_noHard_T2"
RESUlTS_FOLDER_NAME3_T2="UNet2d5_Att_Hard_T2"

cd $PROJECT_PATH

python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME1_T1 --dataset T1 --no_attention --no_hardness 2> train_error_log1_T1.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME2_T1 --dataset T1                --no_hardness 2> train_error_log2_T1.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME3_T1 --dataset T1                              2> train_error_log3_T1.txt &

sleep 60 # to prevent gpu from running out of memory

python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME1_T2 --dataset T2 --no_attention --no_hardness 2> train_error_log1_T2.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME2_T2 --dataset T2                --no_hardness 2> train_error_log2_T2.txt &
python3 VS_train.py --results_folder_name $RESUlTS_FOLDER_NAME3_T2 --dataset T2                              2> train_error_log3_T2.txt

python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME1_T1 --dataset T1 --no_attention --no_hardness 2> inference_error_log1_T1.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME2_T1 --dataset T1                --no_hardness 2> inference_error_log2_T1.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME3_T1 --dataset T1                              2> inference_error_log3_T1.txt

python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME1_T2 --dataset T2 --no_attention --no_hardness 2> inference_error_log1_T2.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME2_T2 --dataset T2                --no_hardness 2> inference_error_log2_T2.txt
python3 VS_inference.py --results_folder_name $RESUlTS_FOLDER_NAME3_T2 --dataset T2                              2> inference_error_log3_T2.txt