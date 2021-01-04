#!/usr/bin/env python
# coding: utf-8

import argparse
import monai

from params.VSparams import VSparams

monai.config.print_config()

# read parsed arguments
parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = VSparams(parser)

# create folders
p.create_results_folders()

# set up logger
logger = p.set_up_logger("training_log.txt")

# log parameters
p.log_parameters()

# load paths to data sets
train_files, val_files, test_files = p.load_T1_or_T2_data()

# define the transforms
train_transforms, val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# check transforms
p.check_transforms_on_first_validation_image_and_label(val_files, val_transforms)

# cache and load data
train_loader = p.cache_transformed_train_data(train_files, train_transforms)
val_loader = p.cache_transformed_val_data(val_files, val_transforms)

# create UNet, DiceLoss and Adam optimizer
model = p.set_and_get_model()
loss_function = p.set_and_get_loss_function()
optimizer = p.set_and_get_optimizer(model)

# run training algorithm
epoch_loss_values, metric_values = p.run_training_algorithm(model, loss_function, optimizer, train_loader, val_loader)

# plot loss and mean dice
p.plot_loss_curve_and_mean_dice(epoch_loss_values, metric_values)
