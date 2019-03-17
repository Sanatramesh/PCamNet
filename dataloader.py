import os
import pickle
import numpy as np
from termcolor import colored

from torch.utils.data import Dataset, DataLoader


class CIFAR10Loader(Dataset):
    '''Data loader for cifar10 dataset'''

    def __init__(self, data_path='data/cifar-10-batches-py', transform=None):
        self.data_path = data_path
        self.transform = transform

        self.data = []
        self.labels = []
        self.init_loader()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_train = self.data[idx]
        label_train  = self.labels[idx]

        if self.transform:
            sample_train = self.transform(sample_train)
            label_train  = self.transform(label_train)

        return sample_train, label_train

    def init_loader(self, type='train'):
        if type == 'train':
            batch_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
        elif type == 'test':
            batch_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]

        for batch in batch_list:
            print(colored('====> ', 'blue') + 'Processing file: ', os.path.join(self.data_path, batch[0]))
            batch = unpickle(os.path.join(self.data_path, batch[0]))
            tmp = batch[b'data']
            self.data.append(tmp)
            self.labels.append(batch[b'labels'])

        self.data = np.float32(np.concatenate(self.data))
        self.data = self.data.reshape(self.data.shape[0], 3, 32, 32) #.swapaxes(1, 3).swapaxes(1, 2)

        self.labels = np.concatenate(self.labels).astype(np.long)
        
        print('Data dims, Label dims :', self.data.shape, self.labels.shape)


def unpickle(file):
    with open(file, 'rb') as fp:
        dict = pickle.load(fp)

    return dict


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

        return sample_train, label_train
