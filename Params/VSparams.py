import os
import logging
import numpy as np
from natsort import natsorted
from time import perf_counter
import glob
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, NormalizeIntensityd, SpatialPadd, RandFlipd, RandSpatialCropd, Orientationd, \
    ToTensord, CenterSpatialCropd
from monai.networks.layers import Norm
# from torchviz import make_dot
# import hiddenlayer as hl
from .networks.nets.unet2d5 import UNet2d5

monai.config.print_config()


class VSparams:

    def __init__(self, args):
        self.dataset = 'T2'  # choose 'T1' or 'T2'
        self.data_root = './data/VS_crop/'  # set path to data set
        self.num_train, self.num_val, self.num_test = 198, 10, 40  # number of images in training, validation and test set AFTER discarding
        self.discard_cases_idx = [219]  # specify indices of cases that are discarded
        self.pad_crop_shape = [128, 128, 32]
        self.pad_crop_shape_test = [256, 128, 32]
        self.num_workers = 4
        self.torch_device_arg = 'cuda:0'
        if hasattr(args, 'train_batch_size'):
            self.train_batch_size = args.train_batch_size
        else:
            self.train_batch_size = None
        if hasattr(args, 'initial_learning_rate'):
            self.initial_learning_rate = args.initial_learning_rate
        else:
            self.initial_learning_rate = None
        self.epochs_with_const_lr = 100
        self.weight_decay = 1e-7
        self.num_epochs = 300
        self.val_interval = 2  # determines how frequently validation is performed during training
        self.model = "UNet2d5"

        # paths
        self.results_folder_path = os.path.join(self.data_root, 'results', args.results_folder_name)
        self.logs_path = os.path.join(self.results_folder_path, 'logs')
        self.model_path = os.path.join(self.results_folder_path, 'model')
        self.figures_path = os.path.join(self.results_folder_path, 'figures')

        #
        self.device = torch.device(self.torch_device_arg)

    def create_results_folders(self):
        # create results folders for logs, figures and model
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path, exist_ok=False)
            os.chmod(self.logs_path, 0o777)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=False)
            os.chmod(self.model_path, 0o777)
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path, exist_ok=False)
            os.chmod(self.figures_path, 0o777)

    def set_up_logger(self, log_file_name):
        # logging settings
        self.logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.logs_path, log_file_name), mode='w')
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)
        formatter = logging.Formatter('%(asctime)s %(levelname)s        %(message)s')
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Created " + log_file_name)
        return self.logger

    def log_parameters(self):
        logger = self.logger
        # write all parameters to log
        logger.info('-' * 10)
        logger.info("Parameters: ")
        logger.info('dataset =                      {}'.format(self.dataset))
        logger.info('data_root =                    {}'.format(self.data_root))
        logger.info('num_train, num_val, num_test = {}, {}, {}'.format(self.num_train, self.num_val, self.num_test))
        logger.info('discard_cases_idx =            {}'.format(self.discard_cases_idx))
        logger.info('pad_crop_shape =               {}'.format(self.pad_crop_shape))
        logger.info('pad_crop_shape_test =          {}'.format(self.pad_crop_shape_test))
        logger.info('num_workers =                  {}'.format(self.num_workers))
        logger.info('torch_device_arg =             {}'.format(self.torch_device_arg))
        logger.info('train_batch_size =             {}'.format(self.train_batch_size))
        logger.info('initial_learning_rate =        {}'.format(self.initial_learning_rate))
        logger.info('epochs_with_const_lr =         {}'.format(self.epochs_with_const_lr))
        logger.info('weight_decay =                 {}'.format(self.weight_decay))
        logger.info('num_epochs =                   {}'.format(self.num_epochs))
        logger.info('model =                        {}'.format(self.model))

        logger.info('results_folder_path =          {}'.format(self.results_folder_path))
        logger.info('-' * 10)

    def load_T1_or_T2_data(self):
        logger = self.logger
        # find and sort all files under the following paths
        if self.dataset == 'T1':
            logger.info('Load T1 data set')
            all_images = natsorted(glob.glob(os.path.join(self.data_root, 'VS_T1_crop', '*t1.nii.gz')))
            all_labels = natsorted(glob.glob(os.path.join(self.data_root, 'VS_T1_crop', '*seg.nii.gz')))
        elif self.dataset == 'T2':
            logger.info('Load T2 data set')
            all_images = natsorted(glob.glob(os.path.join(self.data_root, 'VS_T2_crop', '*t2.nii.gz')))
            all_labels = natsorted(glob.glob(os.path.join(self.data_root, 'VS_T2_crop', '*seg.nii.gz')))
        else:
            raise Exception("The dataset '" + self.dataset + "' is not defined.")

        assert (len(all_images) == len(all_labels)), "Not the same number of images and labels"
        assert (len(all_images) >= sum((self.num_train, self.num_val, self.num_test, len(self.discard_cases_idx)))), \
            "Sum of desired training, validation, test and discarded set size is larger than total number of images in data set"

        # discard cases
        for i in self.discard_cases_idx:
            elimination_str = 'gk_' + str(i) + '_'
            for im_idx, path in enumerate(all_images):
                if elimination_str in path:
                    all_images.pop(im_idx)
                    all_labels.pop(im_idx)

        # create a list of dictionaries, each of which contains an image and a label
        data_dicts = [{'image': image_name, 'label': label_name}
                      for image_name, label_name in zip(all_images, all_labels)]

        # split up the dictionaries into training, validation and test set
        train_files, val_files, test_files = data_dicts[:self.num_train], \
                                             data_dicts[self.num_train:self.num_train + self.num_val], \
                                             data_dicts[self.num_train + self.num_val
                                                        :
                                                        self.num_train + self.num_val + self.num_test]

        logger.info('Number of images in training set   = {}'.format(len(train_files)))
        logger.info('Number of images in validation set = {}'.format(len(val_files)))
        logger.info('Number of images in test set       = {}'.format(len(test_files)))
        logger.info('training set   = {}'.format(train_files))
        logger.info('validation set = {}'.format(val_files))
        logger.info('test set       = {}'.format(test_files))

        # return as dictionaries of image/label pairs
        return train_files, val_files, test_files

    def get_transforms(self):
        # Setup transforms of data sets
        train_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image']),
            SpatialPadd(keys=['image', 'label'], spatial_size=self.pad_crop_shape),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandSpatialCropd(keys=['image', 'label'], roi_size=self.pad_crop_shape, random_center=True,
                             random_size=False),
            ToTensord(keys=['image', 'label'])
        ])
        val_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image']),
            SpatialPadd(keys=['image', 'label'], spatial_size=self.pad_crop_shape),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandSpatialCropd(keys=['image', 'label'], roi_size=self.pad_crop_shape, random_center=True,
                             random_size=False),
            ToTensord(keys=['image', 'label'])
        ])

        test_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image']),
            SpatialPadd(keys=['image', 'label'], spatial_size=self.pad_crop_shape_test),
            CenterSpatialCropd(keys=['image', 'label'], roi_size=self.pad_crop_shape_test),
            ToTensord(keys=['image', 'label'])
        ])

        return train_transforms, val_transforms, test_transforms

    def get_center_of_mass_slice(self, label):
        # calculate center of mass of label in trough plan direction to select a slice that shows the tumour
        num_slices = label.shape[2]
        slice_masses = np.zeros(num_slices)
        for z in range(num_slices):
            slice_masses[z] = label[:, :, z].sum()

        if sum(slice_masses) == 0:  # if there is no label in the cropped image
            slice_weights = np.ones(num_slices) / num_slices  # give all slices equal weight
        else:
            slice_weights = slice_masses / sum(slice_masses)

        center_of_mass = sum(slice_weights * np.arange(num_slices))
        slice_closest_to_center_of_mass = int(center_of_mass.round())
        return slice_closest_to_center_of_mass

    def check_transforms_on_first_validation_image_and_label(self, val_files, val_transforms):
        logger = self.logger
        # check the transforms on the first validation set image
        check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)  # apply transform
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = monai.utils.misc.first(check_loader)  # gets the first item from an input iterable
        image, label = (check_data['image'][0][0], check_data['label'][0][0])
        logger.info('-' * 10)
        logger.info('Check the transforms on the first validation set image and label')
        logger.info('Length of check_data = {}'.format(
            len(check_data)))  # this dictionary also contains all the nifti header info
        logger.info("check_data['image'].shape = {}".format(check_data['image'].shape))
        logger.info('Validation image shape = {}'.format(image.shape))
        logger.info('Validation label shape = {}'.format(label.shape))

        slice_idx = self.get_center_of_mass_slice(
            label)  # choose slice of selected validation set image volume for the figure

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
        plt.savefig(os.path.join(self.figures_path, 'check_validation_image_and_label.png'))

    # Set different seed for workers of DataLoader
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))

    def cache_transformed_train_data(self, train_files, train_transforms):
        # Define CacheDataset and DataLoader for training and validation
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=self.num_workers)
        train_loader = DataLoader(train_ds, batch_size=self.train_batch_size, shuffle=True,
                                  num_workers=self.num_workers, collate_fn=monai.data.list_data_collate,
                                  worker_init_fn=self.worker_init_fn)
        return train_loader

    def cache_transformed_val_data(self, val_files, val_transforms):
        val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0,
                                         num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=self.num_workers)
        return val_loader

    def cache_transformed_test_data(self, test_files, test_transforms):
        test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0,
                                          num_workers=self.num_workers)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=self.num_workers)
        return test_loader

    def set_and_get_model(self):

        if self.model == "UNet":
            model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2,
                                             channels=(16, 32, 48, 64, 80),
                                             strides=(2, 2, 2, 2),
                                             num_res_units=2, norm=Norm.BATCH).to(self.device)
        elif self.model == "UNet2d5":
            s = 2
            k = 2
            model = UNet2d5(dimensions=3, in_channels=1, out_channels=2,
                            channels=(16, 32, 48, 64, 80),
                            strides=((s, s, 1),
                                     (s, s, 1),
                                     (s, s, s),
                                     (s, s, s),),
                            kernel_sizes=((3, 3, 1),
                                          (3, 3, 1),
                                          (3, 3, 3),
                                          (3, 3, 3),
                                          (3, 3, 3),),
                            sample_kernel_sizes=((k, k, 1),
                                                 (k, k, 1),
                                                 (k, k, k),
                                                 (k, k, k),),
                            num_res_units=2,
                            norm=Norm.BATCH,
                            dropout=0.1
                            ).to(self.device)

        # hl.build_graph(model, torch.zeros(2, 1, 128, 128, 32)).save("model")
        return model

    def set_and_get_loss_function(self):
        loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        return loss_function

    def set_and_get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def compute_dice_score(self, predicted_probabilities, label):
        n_classes = predicted_probabilities.shape[1]
        y_pred = torch.argmax(predicted_probabilities, dim=1, keepdim=True)  # pick larger value of 2 channels
        y_pred = monai.networks.utils.one_hot(y_pred, n_classes)  # make 2 channel one hot tensor
        dice_score = torch.tensor(
            [[1 - monai.losses.DiceLoss(include_background=False, to_onehot_y=True, softmax=False,
                                        reduction="mean").forward(y_pred, label, smooth=1e-5)]], device=self.device)
        return dice_score

    def run_training_algorithm(self, model, loss_function, optimizer, train_loader, val_loader):
        logger = self.logger
        epochs_with_const_lr = self.epochs_with_const_lr
        val_interval = self.val_interval  # validation every val_interval epochs

        # Execute training process
        best_metric = -1  # stores highest mean Dice score obtained during validation
        best_metric_epoch = -1  # stores the epoch number during which the highest mean Dice score was obtained
        epoch_loss_values = list()  # stores losses of every epoch
        metric_values = list()  # stores Dice scores of every val_interval epoch
        num_epochs = self.num_epochs
        start = perf_counter()
        for epoch in range(num_epochs):
            logger.info('-' * 10)
            logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
            if epoch == val_interval:
                stop = perf_counter()
                logger.info(('Average duration of first {0:.0f} epochs = {1:.2f} s. ' +
                             'Expected total training time = {2:.2f} h')
                            .format(val_interval, (stop - start) / val_interval,
                                    (stop - start) * num_epochs / val_interval / 3600))
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
                optimizer.zero_grad()  # reset the optimizer gradient
                outputs = model(inputs)  # evaluate the model
                # make_dot(outputs.mean(), params=dict(model.named_parameters())).render("attached", format="png")
                loss = loss_function(outputs, labels)
                loss.backward()  # computes the gradients
                optimizer.step()  # update the model weights
                epoch_loss += loss.item()
                if epoch == 0:
                    logger.info(
                        '{}/{}, train_loss: {:.4f}'.format(step, self.num_train // train_loader.batch_size,
                                                           loss.item()))
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
                        val_inputs, val_labels = val_data['image'].to(self.device), val_data['label'].to(self.device)
                        val_outputs = model(val_inputs)

                        # value1 = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                        #                           to_onehot_y=True, mutually_exclusive=True)

                        dice_score = self.compute_dice_score(val_outputs, val_labels)

                        metric_count += len(dice_score)
                        metric_sum += dice_score.sum().item()
                    metric = metric_sum / metric_count  # calculate mean Dice score of current epoch for validation set
                    if metric > best_metric:  # if it's the best Dice score so far, proceed to save
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        # save the current best model weights
                        torch.save(model.state_dict(), os.path.join(self.model_path, 'best_metric_model.pth'))
                        logger.info('saved new best metric model')
                    logger.info('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                        epoch + 1, metric, best_metric, best_metric_epoch))

            # learning rate update
            if (epoch + 1) % epochs_with_const_lr == 0:
                for param_group in optimizer.param_groups:
                    lr_divisor = 10.0
                    param_group['lr'] = param_group['lr'] / lr_divisor
                    logger.info('Dividing learning rate by {}. '
                                'New learning rate is: lr = {}'.format(lr_divisor, param_group['lr']))

        logger.info('Train completed, best_metric: {:.4f}  at epoch: {}'.format(best_metric, best_metric_epoch))
        return epoch_loss_values, metric_values

    def plot_loss_curve_and_mean_dice(self, epoch_loss_values, metric_values):
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
        x = [self.val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel('epoch')
        plt.plot(x, y)
        plt.savefig(os.path.join(self.figures_path, 'epoch_average_loss_and_val_mean_dice.png'))

    def load_trained_state_of_model(self, model):
        # load the trained model and set it into evaluation mode
        model.load_state_dict(torch.load(os.path.join(self.model_path, 'best_metric_model.pth')))
        return model

    def run_inference(self, model, data_loader):
        logger = self.logger
        model.eval()  # activate evaluation mode of model
        dice_scores = np.zeros(len(data_loader))
        with torch.no_grad():  # turns of PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
                logger.info('starting image {}'.format(i))
                outputs = model(data['image'].to(self.device))

                dice_score = self.compute_dice_score(outputs, data['label'].to(self.device))
                dice_scores[i] = dice_score.item()

                logger.info(f"dice_score = {dice_score.item()}")

                # plot centre of mass slice of label
                label = torch.squeeze(data['label'][0, 0, :, :, :])
                slice_idx = self.get_center_of_mass_slice(
                    label)  # choose slice of selected validation set image volume for the figure
                plt.figure('check', (18, 6))
                plt.subplot(1, 3, 1)
                plt.title('image ' + str(i) + ', slice = ' + str(slice_idx))
                plt.imshow(data['image'][0, 0, :, :, slice_idx], cmap='gray')
                plt.subplot(1, 3, 2)
                plt.title('label ' + str(i))
                plt.imshow(data['label'][0, 0, :, :, slice_idx])
                plt.subplot(1, 3, 3)
                plt.title('output ' + str(i))
                plt.imshow(torch.argmax(outputs, dim=1).detach().cpu()[0, :, :, slice_idx])
                plt.savefig(os.path.join(self.figures_path, 'best_model_output_val' + str(i) + '.png'))

        print(f"all_dice_scores = {dice_scores}")
        print(f"mean_dice_score = {dice_scores.mean()} +- {dice_scores.std()}")
