import os
import configparser
from comm.registry import Registry
from ast import literal_eval

from compute import Config_ini


class Config(object):

    def __init__(self, config_file, section):
        self.config_file = config_file
        self.section = section

    def write_ini_file(self, key, value):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        config.set(self.section, key, value)
        config.write(open(self.config_file, 'w'))

    def read_ini_file(self, key):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config.get(self.section, key)

    def read_ini_file_all(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config[self.section]


class TrainConfig(object):

    def __init__(self):
        pass

    @staticmethod
    def ConfigTrainModel(trainModel):
        """Configurate the training model
        """
        # load the dataset
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        dataset = Config_ini.dataset
        if dataset in ls_dataset:
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        train_loader = dataloader_cls_ins.get_train_dataloader()
        valid_loader = dataloader_cls_ins.get_val_dataloader()
        test_loader = dataloader_cls_ins.get_test_dataloader()

        trainModel.batch_size = int(Config_ini.batch_size)
        trainModel.nepochs = int(Config_ini.total_epoch)
        trainModel.lr = float(Config_ini.lr)

        trainModel.trainloader = train_loader
        trainModel.validate_loader = valid_loader
        trainModel.testloader = test_loader

    @staticmethod
    def get_out_cls_num(dataset):
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        if dataset in ls_dataset:
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        return dataloader_cls_ins.out_cls_num

    @staticmethod
    def get_data_input_size(dataset):
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        if dataset in ls_dataset:
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        return dataloader_cls_ins.input_size


