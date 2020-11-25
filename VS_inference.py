#!/usr/bin/env python
# coding: utf-8

import argparse
import monai

from params.VSparams import VSparams

monai.config.print_config()

# read parsed arguments
parser = argparse.ArgumentParser(description='Train the model')

# initialize parameters
p = VSparams(parser)

# set up logger
logger = p.set_up_logger('test_log.txt')

# log parameters
p.log_parameters()

# load paths to data sets
train_files, val_files, test_files = p.load_T1_or_T2_data()

# define the transforms
train_transforms, val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# cache and load validation data
test_loader = p.cache_transformed_test_data(test_files, test_transforms)

# create UNet
model = p.set_and_get_model()

# load the trained state of the model
model = p.load_trained_state_of_model(model)

# run inference and create figures in figures folder
p.run_inference(model, test_loader)
