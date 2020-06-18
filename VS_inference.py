#!/usr/bin/env python
# coding: utf-8

import os
import logging
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, Spacingd, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd, Orientationd, ToTensord
from monai.data import list_data_collate
from monai.networks.layers import Norm
monai.config.print_config()

dataset = 'T2'  # choose 'T1' or 'T2'
data_root = './data/VS_crop/'  # set path to data set
num_train, num_val, num_test = 178, 20, 47  # number of images in training, validation and test set
pad_crop_shape = [384, 160, 176]
spacing_pix_dim = (0.4, 0.4, 0.4)
num_workers = 4
torch_device_arg = 'cuda:0'

# logging settings
logger = logging.getLogger('VS_training')
fileHandler = logging.FileHandler(os.path.join(data_root, 'results', 'validation_log.txt'), mode='w')
consoleHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.info("Created validation_log.txt")

# find and sort all files under the following paths
if dataset == 'T1':
    logger.info('Load T1 data set')
    all_images = sorted(glob.glob(os.path.join(data_root, 'VS_T1_crop', '*t1.nii.gz')))
    all_labels = sorted(glob.glob(os.path.join(data_root, 'VS_T1_crop', '*seg.nii.gz')))
elif dataset == 'T2':
    logger.info('Load T2 data set')
    all_images = sorted(glob.glob(os.path.join(data_root, 'VS_T2_crop', '*t2.nii.gz')))
    all_labels = sorted(glob.glob(os.path.join(data_root, 'VS_T2_crop', '*seg.nii.gz')))

assert(len(all_images) == len(all_labels)), "Not the same number of images and labels"

# create a list of dictionaries, each of which contains an image and a label
data_dicts = [{'image': image_name, 'label': label_name}
              for image_name, label_name in zip(all_images, all_labels)]

# randomly split up the dictionaries into training, validation and test set
random.seed(0)
random.shuffle(data_dicts)
train_files, val_files, test_files = data_dicts[:num_train], \
                                     data_dicts[num_train:num_train+num_val], \
                                     data_dicts[num_train+num_val:num_train+num_val+num_test]
logger.info('Number of images in training set = {}'.format(len(train_files)))
logger.info('Number of images in validation set = {}'.format(len(val_files)))
logger.info('training set   = {}'.format(train_files))
logger.info('validation set = {}'.format(train_files))
logger.info('test set       = {}'.format(train_files))

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
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers
)

# create UNet, DiceLoss and Adam optimizer
device = torch.device(torch_device_arg)
model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)

# load the cached validation data set
val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

# load the trained model and set it into evaluation mode
model.load_state_dict(torch.load('best_metric_model.pth'))
model.eval()

# select slice for plots
slice_idx = 16
with torch.no_grad():  # turns of PyTorch's auto grad for better performance
    for i, val_data in enumerate(val_loader):
        logger.info('starting forward pass validation image {}'.format(i))
        val_outputs = model(val_data['image'].to(device))
        # plot the slice [:, :, slice_idx]
        plt.figure('check', (18, 6))
        plt.subplot(1, 3, 1)
        plt.title('image ' + str(i))
        plt.imshow(val_data['image'][0, 0, :, :, slice_idx], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('label ' + str(i))
        plt.imshow(val_data['label'][0, 0, :, :, slice_idx])
        plt.subplot(1, 3, 3)
        plt.title('output ' + str(i))
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_idx])
        plt.savefig(os.path.join(data_root, 'results', 'best_model_output_val' + str(i) + '.png'))