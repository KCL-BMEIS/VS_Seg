#!/usr/bin/env python
# coding: utf-8

import os
import logging
import argparse
from natsort import natsorted
from time import strftime
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, Spacingd, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd, Orientationd, \
    ToTensord
from monai.data import list_data_collate
from monai.networks.layers import Norm

monai.config.print_config()

# read and configure arguments
parser = argparse.ArgumentParser(description='Evaluate a trained model')
parser.add_argument('--results_folder_name', type=str, default='temp' + strftime("%Y%m%d%H%M%S"),
                    help='name of results folder')
args = parser.parse_args()

dataset = 'T2'  # choose 'T1' or 'T2'
data_root = './data/VS_crop/'  # set path to data set
results_folder_path = os.path.join(data_root, 'results', args.results_folder_name)
num_train, num_val, num_test = 198, 10, 40  # number of images in training, validation and test set
discard_cases_idx = [219]  # specify indices of cases that are discarded
pad_crop_shape = [384, 160, 176]
spacing_pix_dim = (0.4, 0.4, 0.4)
num_workers = 4
torch_device_arg = 'cuda:0'

# set up paths
logs_path = os.path.join(results_folder_path, 'logs')
figures_path = os.path.join(results_folder_path, 'figures')
model_path = os.path.join(results_folder_path, 'model')

# logging settings
logger = logging.getLogger('VS_validation')
fileHandler = logging.FileHandler(os.path.join(logs_path, 'validation_log.txt'), mode='w')
consoleHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.info("Created validation_log.txt")

# write all parameters to log
logger.info('-'*10)
logger.info("Parameters: ")
logger.info('dataset =                      {}'.format(dataset))
logger.info('data_root =                    {}'.format(data_root))
logger.info('results_folder_path =          {}'.format(results_folder_path))
logger.info('num_train, num_val, num_test = {}, {}, {}'.format(num_train, num_val, num_test))
logger.info('discard_cases_idx =            {}'.format(discard_cases_idx))
logger.info('pad_crop_shape =               {}'.format(pad_crop_shape))
logger.info('spacing_pix_dim =              {}'.format(spacing_pix_dim))
logger.info('num_workers =                  {}'.format(num_workers))
logger.info('torch_device_arg =             {}'.format(torch_device_arg))
logger.info('-'*10)

# find and sort all files under the following paths
if dataset == 'T1':
    logger.info('Load T1 data set')
    all_images = natsorted(glob.glob(os.path.join(data_root, 'VS_T1_crop', '*t1.nii.gz')))
    all_labels = natsorted(glob.glob(os.path.join(data_root, 'VS_T1_crop', '*seg.nii.gz')))
elif dataset == 'T2':
    logger.info('Load T2 data set')
    all_images = natsorted(glob.glob(os.path.join(data_root, 'VS_T2_crop', '*t2.nii.gz')))
    all_labels = natsorted(glob.glob(os.path.join(data_root, 'VS_T2_crop', '*seg.nii.gz')))
else:
    raise Exception("The dataset '" + dataset + "' is not defined.")

assert(len(all_images) == len(all_labels)), "Not the same number of images and labels"
assert(len(all_images) >= sum((num_train, num_val, num_test, len(discard_cases_idx)))), \
    "Sum of desired training, validation, test and discarded set size is larger than total number of images in data set"

# discard cases
for i in discard_cases_idx:
    elimination_str = 'gk_' + str(i) + '_'
    for im_idx, path in enumerate(all_images):
        if elimination_str in path:
            all_images.pop(im_idx)
            all_labels.pop(im_idx)

# create a list of dictionaries, each of which contains an image and a label
data_dicts = [{'image': image_name, 'label': label_name}
              for image_name, label_name in zip(all_images, all_labels)]

# split up the dictionaries into training, validation and test set
train_files, val_files, test_files = data_dicts[:num_train], \
                                     data_dicts[num_train:num_train+num_val], \
                                     data_dicts[num_train+num_val:num_train+num_val+num_test]

logger.info('Number of images in training set   = {}'.format(len(train_files)))
logger.info('Number of images in validation set = {}'.format(len(val_files)))
logger.info('Number of images in test set       = {}'.format(len(test_files)))
logger.info('training set   = {}'.format(train_files))
logger.info('validation set = {}'.format(val_files))
logger.info('test set       = {}'.format(test_files))

# Setup transforms of data sets
train_transforms = Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Orientationd(keys=['image', 'label'], axcodes='RAS'),
    NormalizeIntensityd(keys=['image']),
    Spacingd(keys=['image', 'label'], pixdim=spacing_pix_dim, interp_order=(3, 0), mode='nearest'),
    SpatialPadd(keys=['image', 'label'], spatial_size=pad_crop_shape),
    CenterSpatialCropd(keys=['image', 'label'], roi_size=pad_crop_shape),
    ToTensord(keys=['image', 'label'])
])
val_transforms = Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Orientationd(keys=['image', 'label'], axcodes='RAS'),
    NormalizeIntensityd(keys=['image']),
    Spacingd(keys=['image', 'label'], pixdim=spacing_pix_dim, interp_order=(3, 0), mode='nearest'),
    SpatialPadd(keys=['image', 'label'], spatial_size=pad_crop_shape),
    CenterSpatialCropd(keys=['image', 'label'], roi_size=pad_crop_shape),
    ToTensord(keys=['image', 'label'])
])

# load the data, transform it, and cache the transformed data
val_ds = monai.data.CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

# create UNet
device = torch.device(torch_device_arg)
model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)

# load the trained model and set it into evaluation mode
model.load_state_dict(torch.load(os.path.join(model_path, 'best_metric_model.pth')))
model.eval()

with torch.no_grad():  # turns of PyTorch's auto grad for better performance
    for i, val_data in enumerate(val_loader):
        logger.info('starting forward pass validation image {}'.format(i))
        val_outputs = model(val_data['image'].to(device))

        label = torch.squeeze(val_data['label'][0, 0, :, :, :])
        # calculate center of mass of label in trough plan direction to select a slice that shows the tumour
        num_slices = label.shape[2]
        slice_masses = np.zeros(num_slices)
        for z in range(num_slices):
            slice_masses[z] = label[:, :, z].sum()

        slice_weights = slice_masses / sum(slice_masses)
        center_of_mass = sum(slice_weights * np.arange(num_slices))
        slice_closest_to_center_of_mass = int(center_of_mass.round())

        slice_idx = slice_closest_to_center_of_mass  # choose slice of selected validation set image volume for the figure
        plt.figure('check', (18, 6))
        plt.subplot(1, 3, 1)
        plt.title('image ' + str(i) + ', slice = ' + str(slice_closest_to_center_of_mass))
        plt.imshow(val_data['image'][0, 0, :, :, slice_idx], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('label ' + str(i))
        plt.imshow(val_data['label'][0, 0, :, :, slice_idx])
        plt.subplot(1, 3, 3)
        plt.title('output ' + str(i))
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_idx])
        plt.savefig(os.path.join(figures_path, 'best_model_output_val' + str(i) + '.png'))