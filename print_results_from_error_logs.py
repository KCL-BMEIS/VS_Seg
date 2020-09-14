import os

results_path = "/mnt/dgx-server/projects/VS_Seg"

file_names = ["train_error_log1_T1.txt",
              "train_error_log2_T1.txt",
              "train_error_log3_T1.txt",
              "train_error_log1_T2.txt",
              "train_error_log2_T2.txt",
              "train_error_log3_T2.txt",
              "inference_error_log1_T1.txt",
              "inference_error_log2_T1.txt",
              "inference_error_log3_T1.txt",
              "inference_error_log1_T2.txt",
              "inference_error_log2_T2.txt",
              "inference_error_log3_T2.txt",
              ]

for file_name in file_names:
    with open(os.path.join(results_path, file_name)) as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        print(last_line)
