# coding=utf-8
import os

from compute import Config_ini


class BaseDataloader(object):
    """Abstract class for Dataloader
    """

    def __init__(self):
        self.input_size = None  # shape of the input for DNN
        self.out_cls_num = None  # number of classes for classification
        self.root = None  # root path of the data
        self.download = True  # download the data

        self.augment = True  # flag of data augmentation
        self.valid_size = None  # ratio betweem number of training data and that of test data.
        self.shuffle = True  # shuffle the dataset
        self.random_seed = 2312390  # the random seed
        self.show_sample = False  # display the samples
        self.num_workers = 1  # number of workers
        self.pin_memory = True
        self.data_dir = os.path.join('.', Config_ini.data_dir.split('/')[-1])

        self.img_input_size = Config_ini.img_input_size
        self.batch_size = Config_ini.batch_size
        self.nepochs = Config_ini.total_epoch
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def get_train_dataloader(self):
        raise NotImplementedError()

    def get_val_dataloader(self):
        raise NotImplementedError()

    def get_test_dataloader(self):
        raise NotImplementedError()

    def amend_valid_size(self, val):
        self.valid_size = val
