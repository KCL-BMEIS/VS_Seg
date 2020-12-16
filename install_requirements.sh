#!/bin/bash
export HOME=/workspace/home/packages/
export PATH=$HOME/.local/bin:$PATH
pip install git+https://github.com/Project-MONAI/MONAI.git@0b415f09db82a39f1df5bafd9b8ff67f9f3f677c
pip install nibabel
pip install natsort
pip install tqdm --upgrade