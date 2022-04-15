# coding=utf-8


class BaseDataloader(object):
    """Abstract class for Dataloader
    """

    def __init__(self):
        self.input_size = None  # shape of the input for DNN
        self.out_cls_num = None  # number of classes for classification
        self.root = None  # root path of the data
        self.download = True  # download the data?

        self.augment = True  # flag of data augmentation
        self.valid_size = 0.2  # ratio betweem number of training data and that of test data.
        self.shuffle = True  # shuffle the dataset
        self.random_seed = 2312390  # the random seed
        self.show_sample = False  # display the samples
        self.num_workers = 0  # number of workers
        self.pin_memory = True

        from train.utils import OptimizerConfig
        from train.utils import DatasetConfig

        opt_config = OptimizerConfig()
        self.batch_size = int(opt_config.read_ini_file('_batch_size'))
        self.nepochs = int(opt_config.read_ini_file('_total_epoch'))

        data_config = DatasetConfig()
        self.data_dir = data_config.read_ini_file('_dir')
        self.img_input_size = [int(_id) for _id in data_config.read_ini_file('_input_size').split(',')]
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

    def get_train_dataloader(self):
        raise NotImplementedError()

    def get_val_dataloader(self):
        raise NotImplementedError()

    def get_test_dataloader(self):
        raise NotImplementedError()
