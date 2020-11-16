#!/bin/bash

pip install git+https://github.com/Project-MONAI/MONAI.git@0d197e6bea9dd2244c63b80a80b464ef23a5aab9
pip install nibabel
pip install natsort
apt -y update
apt -y install tmux

