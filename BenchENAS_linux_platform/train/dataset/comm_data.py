import math
import os
import pathlib
import random
from abc import ABC

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from train.dataset.dataloader import BaseDataloader


class FDataLoader(BaseDataloader):
    def __init__(self):
        super(FDataLoader, self).__init__()
        self.root = self.data_dir
        dir_list = os.listdir(self.root)
        self.train_test = 0
        if 'train' in dir_list and 'test' in dir_list:
            self.train_test = 1
        elif 'train' in dir_list:
            self.train_test = 2
        else:
            print('Not excepted dataset')
        self.input_size = self.img_input_size
        self.out_cls_num = os.listdir(os.path.join(self.root, 'train'))

    def get_train_dataloader(self):
        if self.train_test == 1:
            self.train_dataloader = self.__get_train_loader(self.root,
                                                            self.batch_size,
                                                            self.shuffle,
                                                            self.show_sample,
                                                            self.num_workers,
                                                            self.pin_memory
                                                            )
        else:
            self.train_dataloader, self.val_dataloader = self.__get_train_test_loader(self.root,
                                                                                      self.batch_size,
                                                                                      self.augment,
                                                                                      self.random_seed,
                                                                                      self.valid_size,
                                                                                      self.shuffle)
        return self.train_dataloader

    def get_val_dataloader(self):
        if self.val_dataloader is None:
            self.val_dataloader = self.__get_test_loader(self.root,
                                                         self.batch_size,
                                                         self.shuffle,
                                                         self.show_sample,
                                                         self.num_workers,
                                                         self.pin_memory)
        return self.val_dataloader

    def get_test_dataloader(self):
        return self.__get_test_loader(self.root,
                                      self.batch_size,
                                      self.augment,
                                      self.random_seed,
                                      self.shuffle,
                                      self.show_sample)

    def __get_train_test_loader(self, root, batch_size, augment, random_seed, valid_size, shuffle):
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        train_dir = os.path.join(self.root, 'train')
        train_set = DataDeal(train_dir, self.input_size)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            random.shuffle(indices)

        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler)
        test_loader = DataLoader(train_set, batch_size=batch_size,
                                 sampler=test_sampler)

        return train_loader, test_loader

    def __get_train_loader(self, root, batch_size, shuffle, show_sample,
                           num_workers=1, pin_memory=False):
        train_dir = os.path.join(self.root, 'train')
        train_set = DataDeal(train_dir, self.input_size)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

        return train_loader

    def __get_test_loader(self, root, batch_size, shuffle, show_sample,
                          num_workers=1, pin_memory=False):
        test_dir = os.path.join(self.root, 'test')
        test_set = DataDeal(test_dir, self.input_size)

        test_loader = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers)

        return test_loader


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class DataDeal(Dataset):
    def __init__(self, path, input_size):
        data_root = pathlib.Path(path)
        img_suffix = ['.jpg', '.png', '.jpeg', '.gif', '.JPG', '.PNG', '.JPEG', '.GIF']
        all_image_paths = list()
        for i in img_suffix:
            img_name = '*/*' + i
            all_image_paths += list(data_root.glob(img_name))
        self.all_image_paths = [str(path) for path in all_image_paths]
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        self.all_image_labels = [label_to_index[path.parent.name] for path in all_image_paths]
        self.weight = int(input_size[0])
        self.height = int(input_size[1])
        self.mean = np.array(mean).reshape((1, 1, 3))
        self.std = np.array(std).reshape((1, 1, 3))

    def __getitem__(self, index):
        img = cv2.imread(self.all_image_paths[index])
        img = cv2.resize(img, (self.weight, self.height))
        img = img / float(self.weight + 1)
        img = (img - self.mean) / self.std
        img = np.transpose(img, [2, 0, 1])
        label = self.all_image_labels[index]
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)


# f = FDataLoader()
# data = f.get_test_dataloader()
# for img1, label1 in data:
#     print(img1.shape, label1.shape)
#     print(label1)
#     break
