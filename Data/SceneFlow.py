import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from .SceneFlow_helper import read_sceneflow
from .pfm_helper import read_pfm

import pdb

class SceneFlowDataset(Dataset):

    def __init__(self, data_root, npy_root, val_split, test_split, transform, phase):

        super(SceneFlowDataset, self).__init__()

        self.data_root = data_root
        self.npy_root = npy_root
        self.phase = phase
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform

        self.left_imgs, self.right_imgs, self.disps, self.test_left_imgs, self.test_right_imgs, self.test_disps = read_sceneflow(self.data_root)
        #pdb.set_trace()

        assert len(self.left_imgs) == len(self.right_imgs) == len(self.disps), 'Invalid training dataset!'
        assert len(self.test_left_imgs) == len(self.test_right_imgs) == len(self.test_disps), 'Invalid testing dataset!'

        #total_data_num = len(self.left_imgs)
        
        #self.nb_train = int((1 - self.val_split - self.test_split) * total_data_num)
        #self.nb_val = int(self.val_split * total_data_num)
        #self.nb_test = int(self.test_split * total_data_num)

        test_data_num = len(self.test_left_imgs)
        
        self.nb_train = len(self.left_imgs)
        self.nb_val = int(self.val_split * test_data_num)
        self.nb_test = test_data_num
        
        train_npy = os.path.join(self.npy_root, 'train.npy')
        val_npy = os.path.join(self.npy_root, 'val.npy')
        test_npy = os.path.join(self.npy_root, 'test.npy')
        
        if os.path.exists(train_npy) and os.path.exists(val_npy) and os.path.exists(test_npy):
            #self.train_list = np.load(train_npy)
            self.val_list = np.load(val_npy)
            #pdb.set_trace()
            #self.test_list = np.load(test_npy)

            
        else:
            #total_idcs = np.random.permutation(total_data_num)
            #self.train_list = total_idcs[0:self.nb_train]
            #self.val_list = total_idcs[self.nb_train:self.nb_train + self.nb_val]
            #self.test_list = total_idcs[self.nb_train + self.nb_val:]

            test_idcs = np.random.permutation(test_data_num)
            self.val_list = test_idcs[0:self.nb_val]
            
            
            #np.save(train_npy, self.train_list)
            np.save(val_npy, self.val_list)
            #np.save(test_npy, self.test_list)

    def __len__(self):

        if self.phase == 'train':
            return self.nb_train
        elif self.phase == 'val':
            return self.nb_val
        elif self.phase == 'test':
            return self.nb_test


    def __getitem__(self, index):
        
        if self.phase == 'train':
            left_image = self._read_image(self.left_imgs[index])
            right_image = self._read_image(self.right_imgs[index])
            left_disp, scale = read_pfm(self.disps[index])

        elif self.phase == 'val':
            index = self.val_list[index]
            left_image = self._read_image(self.test_left_imgs[index])
            right_image = self._read_image(self.test_right_imgs[index])
            left_disp, scale = read_pfm(self.test_disps[index])

        elif self.phase == 'test':
            left_image = self._read_image(self.test_left_imgs[index])
            right_image = self._read_image(self.test_right_imgs[index])
            left_disp, scale = read_pfm(self.test_disps[index])
            
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        
        left_disp = torch.Tensor(left_disp)
        return left_image, right_image, left_disp, scale

    '''
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
    '''
    
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