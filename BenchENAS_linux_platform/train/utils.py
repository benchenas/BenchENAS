import os
import configparser

from ast import literal_eval


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


class OptimizerConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), 'train.ini')
        Config.__init__(self, file_path, 'optimizer')


class LRConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), 'train.ini')
        Config.__init__(self, file_path, 'LearningRate')


class DatasetConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), 'train.ini')
        Config.__init__(self, file_path, 'dataset')


class TrainConfig(object):

    def __init__(self):
        pass

    @staticmethod
    def ConfigTrainModel(trainModel):
        d = DatasetConfig()
        o = OptimizerConfig()
        l = LRConfig()
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        """Configurate the training model 
        """
        # load the dataset
        dataset = d.read_ini_file('_name')
        if dataset in ls_dataset:
            from comm.registry import Registry
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        train_loader = dataloader_cls_ins.get_train_dataloader()
        valid_loader = dataloader_cls_ins.get_val_dataloader()
        test_loader = dataloader_cls_ins.get_test_dataloader()

        trainModel.batch_size = int(o.read_ini_file('_batch_size'))
        trainModel.nepochs = int(o.read_ini_file('_total_epoch'))
        trainModel.lr = float(l.read_ini_file('lr'))

        trainModel.trainloader = train_loader
        trainModel.validate_loader = valid_loader
        trainModel.testloader = test_loader

    @staticmethod
    def get_out_cls_num():
        d = DatasetConfig()
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        dataset = d.read_ini_file('_name')
        if dataset in ls_dataset:
            from comm.registry import Registry
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        return dataloader_cls_ins.out_cls_num

    @staticmethod
    def get_data_input_size():
        d = DatasetConfig()
        ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
        dataset = d.read_ini_file('_name')
        if dataset in ls_dataset:
            from comm.registry import Registry
            dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
            dataloader_cls_ins = dataloader_cls()
        else:
            from train.dataset.comm_data import FDataLoader
            dataloader_cls_ins = FDataLoader()
        return dataloader_cls_ins.input_size
