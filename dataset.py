import os
import numpy as np
from skimage import io
from sklearn import preprocessing
import torch
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

dataDir = './Images/Patches'
x_trainDir = os.path.join(dataDir,'input_data')
y_trainDir = os.path.join(dataDir,'ref_mask')
mean_fp = './Images/mean.csv'
std_fp = './Images/mean.csv'
mean = np.genfromtxt(mean_fp, delimiter=',')
std = np.genfromtxt(std_fp, delimiter=',')


class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            transform=None):

        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.transform = transform

    def __getitem__(self, i):
        image = io.imread(self.images_fps[i])
        image = np.nan_to_num(image)
        # image = image.transpose(2, 0, 1)

        mask = io.imread(self.masks_fps[i])
        mask = np.nan_to_num(mask)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.transpose(2, 0, 1)

        scaler = preprocessing.MinMaxScaler()
        ascolumns = image.reshape(-1, 3)
        t = scaler.fit_transform(ascolumns)
        image = t.reshape(image.shape)

        m_scaler = preprocessing.MinMaxScaler()
        m_ascolumns = mask.reshape(-1, 1)
        m_t = m_scaler.fit_transform(m_ascolumns)
        mask = m_t.reshape(mask.shape)


        # apply preprocessing
        if self.transform:
            processed = self.transform(image=image, mask=mask)
            # print(processed)
            image = processed['image']
            mask = processed['mask']
            # image = processed

        return image, mask

    def __len__(self):
        return len(self.ids)

# train_dataset = Dataset(
#     x_trainDir,
#     y_trainDir,
#     transform=transformations
# )
#
# image, mask = train_dataset[0]
# print(image.shape)
# print(mask.shape)

#
# smallest = np.amin(image, axis=(0, 1))
# median = np.median(image, axis=(0, 1))
# largest = np.amax(image, axis=(0, 1))
#
# print('smallest: ', smallest)
# print('median: ', median)
# print('largest: ', largest)