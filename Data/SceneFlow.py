import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from .SceneFlow_helper import read_sceneflow
from .pfm_helper import read_pfm

import pdb

class SceneFlowDataset(Dataset):

    def __init__(self, data_root, val_split, test_split, transform, phase):

        super(SceneFlowDataset, self).__init__()

        self.data_root = data_root
        self.phase = phase
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform

        self.left_imgs, self.right_imgs, self.disps, _, _, _ = read_sceneflow(self.data_root)

        assert len(self.left_imgs) == len(self.right_imgs) == len(self.disps), 'Invalid dataset!'

        total_data_num = len(self.left_imgs)

        self.nb_train = int((1 - self.val_split - self.test_split) * total_data_num)
        self.nb_val = int(self.val_split * total_data_num)
        self.nb_test = int(self.test_split * total_data_num)

        if os.path.exists('./Data/train.npy') and os.path.exists('./Data/val.npy') and os.path.exists('./Data/test.npy'):
            self.train_list = np.load('./Data/train.npy')
            self.val_list = np.load('./Data/val.npy')
            self.test_list = np.load('./Data/test.npy')
        else:
            total_idcs = np.random.permutation(total_data_num)
            self.train_list = total_idcs[0:self.nb_train]
            self.val_list = total_idcs[self.nb_train:self.nb_train + self.nb_val]
            self.test_list = total_idcs[self.nb_train + self.nb_val:]
            np.save('./Data/train.npy', self.train_list)
            np.save('./Data/val.npy', self.val_list)
            np.save('./Data/test.npy', self.test_list)

    def __len__(self):

        if self.phase == 'train':
            return self.nb_train
        elif self.phase == 'val':
            return self.nb_val
        elif self.phase == 'test':
            return self.nb_test

    def __getitem__(self, index):

        if self.phase == 'train':
            index = self.train_list[index]
        elif self.phase == 'val':
            index = self.val_list[index]
        elif self.phase == 'test':
            index = self.test_list[index]

        left_image = self._read_image(self.left_imgs[index])
        right_image = self._read_image(self.right_imgs[index])
        left_disp, scale = read_pfm(self.disps[index])

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_disp = torch.Tensor(left_disp)

        return left_image, right_image, left_disp, scale

    def _read_image(self, filename):

        attempt = True
        while attempt:
            try:
                with open(filename, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                attempt = False
            except IOError as e:
                print('[IOError] {}, keep trying...'.format(e))
                attempt = True
        return img