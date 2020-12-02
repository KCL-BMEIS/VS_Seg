#!/bin/bash
export HOME=/workspace/home/packages/
export PATH=$HOME/.local/bin:$PATH
#pip install git+https://github.com/Project-MONAI/MONAI.git@0d197e6bea9dd2244c63b80a80b464ef23a5aab9
pip install monai==0.3.0
pip install nibabel
pip install natsort
pip install tqdm --upgrade
#apt -y update
#apt -y install tmux

