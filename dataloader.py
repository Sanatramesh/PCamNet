import os
import cv2

import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


class CIFAR10Loader(Dataset):
    '''Data loader for cifar10 dataset'''

    def __init__(self, root_dir_img, root_dir_label, transform=None):
        self.root_dir_img = root_dir_img
        self.transform = transform
        self.image_list_train = [f for f in os.listdir(self.root_dir_img) if os.path.isfile(os.path.join(self.root_dir_img, f))]
        self.image_list_train.sort()
        self.root_dir_label = root_dir_label
        self.label_list_train = [f.replace('rs', 'mask') for f in self.image_list_train][1:]
        self.image_list_train = self.image_list_train[1:]

    def __len__(self):
        return len(self.image_list_train)

    def __getitem__(self, idx):
        img_name_train = os.path.join(self.root_dir_img, self.image_list_train[idx])
        label_name_train = os.path.join(self.root_dir_label, self.label_list_train[idx])

        image = cv2.imread(img_name_train)
        label_train = cv2.imread(label_name_train)

        shape = image.shape
        sample_train = image

        label_train = cv2.cvtColor(label_train,cv2.COLOR_BGR2GRAY).reshape(label_train.shape[0],label_train.shape[1],1)
        sample_train = cv2.cvtColor(sample_train,cv2.COLOR_BGR2RGB)

        if self.transform:
            sample_train = self.transform(sample_train)
            label_train = self.transform(label_train)

        return sample_train,label_train


class PCamLoader(Dataset):
    '''Data loader for PCam dataset'''

    def __init__(self, root_dir_img, root_dir_label, transform=None):
        self.root_dir_img = root_dir_img
        self.transform = transform
        self.image_list_train = [f for f in os.listdir(self.root_dir_img) if os.path.isfile(os.path.join(self.root_dir_img, f))]
        self.image_list_train.sort()
        self.root_dir_label = root_dir_label
        self.label_list_train = [f.replace('rs', 'mask') for f in self.image_list_train][1:]
        self.image_list_train = self.image_list_train[1:]

    def __len__(self):
        return len(self.image_list_train)

    def __getitem__(self, idx):
        img_name_train = os.path.join(self.root_dir_img, self.image_list_train[idx])
        label_name_train = os.path.join(self.root_dir_label, self.label_list_train[idx])

        image = cv2.imread(img_name_train)
        label_train = cv2.imread(label_name_train)

        shape = image.shape
        sample_train = image

        label_train = cv2.cvtColor(label_train,cv2.COLOR_BGR2GRAY).reshape(label_train.shape[0],label_train.shape[1],1)
        sample_train = cv2.cvtColor(sample_train,cv2.COLOR_BGR2RGB)

        if self.transform:
            sample_train = self.transform(sample_train)
            label_train = self.transform(label_train)

        return sample_train,label_train
