#!/usr/bin/env python
# coding: utf-8

import os
import logging
import argparse
from natsort import natsorted
from time import perf_counter, strftime
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, Spacingd, SpatialPadd, CenterSpatialCropd, NormalizeIntensityd, Orientationd, \
    ToTensord
from monai.data import list_data_collate
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism

monai.config.print_config()

# read and configure arguments
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--train_batch_size', type=int, default=2, help='batch size of the forward pass')
parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='learning rate at first epoch')
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
train_batch_size = args.train_batch_size
initial_learning_rate = args.initial_learning_rate
epochs_with_const_lr = 100
weight_decay = 1e-7
num_epochs = 600

# create results folders for logs, figures and model
logs_path = os.path.join(results_folder_path, 'logs')
model_path = os.path.join(results_folder_path, 'model')
figures_path = os.path.join(results_folder_path, 'figures')
os.makedirs(logs_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)
os.chmod(logs_path, 0o777)
os.chmod(model_path, 0o777)
os.chmod(figures_path, 0o777)
os.chmod(results_folder_path, 0o777)

# logging settings
logger = logging.getLogger('VS_training')
fileHandler = logging.FileHandler(os.path.join(logs_path, 'training_log.txt'), mode='w')
consoleHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)
formatter = logging.Formatter('%(asctime)s %(levelname)s        %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.info("Created training_log.txt")

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
logger.info('train_batch_size =             {}'.format(train_batch_size))
logger.info('initial_learning_rate =        {}'.format(initial_learning_rate))
logger.info('epochs_with_const_lr =         {}'.format(epochs_with_const_lr))
logger.info('weight_decay =                 {}'.format(weight_decay))
logger.info('num_epochs =                   {}'.format(num_epochs))
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

# Set deterministic training for reproducibility
set_determinism(seed=0)


# Set different seed for workers of DataLoader
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


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

# check the transforms on the first validation set image
check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)  # apply transform
check_loader = DataLoader(check_ds, batch_size=1)
check_data = monai.utils.misc.first(check_loader)  # gets the first item from an input iterable
image, label = (check_data['image'][0][0], check_data['label'][0][0])
logger.info('-' * 10)
logger.info('Check the transforms on the first validation set image and label')
logger.info('Length of check_data = {}'.format(len(check_data)))  # this dictionary also contains all the nifti header info
logger.info("check_data['image'].shape = {}".format(check_data['image'].shape))
logger.info('Validation image shape = {}'.format(image.shape))
logger.info('Validation label shape = {}'.format(label.shape))

# calculate center of mass of label in trough plan direction to select a slice that shows the tumour
num_slices = label.shape[2]
slice_masses = np.zeros(num_slices)
for z in range(num_slices):
    slice_masses[z] = label[:, :, z].sum()

slice_weights = slice_masses/sum(slice_masses)
center_of_mass = sum(slice_weights * np.arange(num_slices))
slice_closest_to_center_of_mass = int(center_of_mass.round())

slice_idx = slice_closest_to_center_of_mass  # choose slice of selected validation set image volume for the figure
logger.info('-' * 10)
logger.info('Plot one slice of the image and the label')
logger.info('image shape: {}, label shape: {}, slice = {}'.format(image.shape, label.shape, slice_idx))
# plot the slice [:, :, slice]
plt.figure('check', (12, 6))
plt.subplot(1, 2, 1)
plt.title('image')
plt.imshow(image[:, :, slice_idx], cmap='gray')
plt.subplot(1, 2, 2)
plt.title('label')
plt.imshow(label[:, :, slice_idx])
plt.savefig(os.path.join(figures_path, 'check_validation_image_and_label.png'))

# Define CacheDataset and DataLoader for training and validation
train_ds = monai.data.CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=list_data_collate, worker_init_fn=worker_init_fn)

val_ds = monai.data.CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

# create UNet, DiceLoss and Adam optimizer
device = torch.device(torch_device_arg)
model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
loss_function = monai.losses.DiceLoss(to_onehot_y=True, do_softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
epochs_with_const_lr = epochs_with_const_lr

# Execute training process
val_interval = 2  # validation every val_interval epochs
best_metric = -1  # stores highest mean Dice score obtained during validation
best_metric_epoch = -1  # stores the epoch number during which the highest mean Dice score was obtained
epoch_loss_values = list()  # stores losses of every epoch
metric_values = list()  # stores Dice scores of every val_interval epoch
num_epochs = num_epochs
start = perf_counter()
for epoch in range(num_epochs):
    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
    if epoch == val_interval:
        stop = perf_counter()
        logger.info(('Average duration of first {0:.0f} epochs = {1:.2f} s. ' +
                     'Expected total training time = {2:.2f} h').format(val_interval,
                                                                        (stop - start) / val_interval,
                                                                        (stop - start) * num_epochs/val_interval/3600))
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()  # reset the optimizer gradient
        outputs = model(inputs)  # evaluate the model
        loss = loss_function(outputs, labels)
        loss.backward()  # computes the gradients
        optimizer.step()  # update the model weights
        epoch_loss += loss.item()
        if epoch == 0:
            logger.info('{}/{}, train_loss: {:.4f}'.format(step, len(train_ds) // train_loader.batch_size, loss.item()))
    epoch_loss /= step  # calculate mean loss over current epoch
    epoch_loss_values.append(epoch_loss)
    logger.info('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

    # validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():  # turns of PyTorch's auto grad for better performance
            metric_sum = 0.
            metric_count = 0
            for val_data in val_loader:  # loop over images in validation set
                val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                val_outputs = model(val_inputs)
                value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                         to_onehot_y=True, mutually_exclusive=True)
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count  # calculate mean Dice score of current epoch for validation set
            metric_values.append(metric)
            if metric > best_metric:  # if it's the best Dice score so far, proceed to save
                best_metric = metric
                best_metric_epoch = epoch + 1
                # save the current best model weights
                torch.save(model.state_dict(), os.path.join(model_path, 'best_metric_model.pth'))
                logger.info('saved new best metric model')
            logger.info('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                epoch + 1, metric, best_metric, best_metric_epoch))

    # learning rate update
    if (epoch + 1) % epochs_with_const_lr == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
            logger.info('Halving learning rate to: lr = {}'.format(param_group['lr']))

logger.info('Train completed, best_metric: {:.4f}  at epoch: {}'.format(best_metric, best_metric_epoch))

# Plot the loss and metric
plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title('Epoch Average Loss')
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title('Val Mean Dice')
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.savefig(os.path.join(figures_path, 'epoch_average_loss_and_val_mean_dice.png'))
