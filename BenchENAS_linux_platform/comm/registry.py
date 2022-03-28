# coding=utf-8

import glob
import importlib
import os

from train.dataset.dataloader import BaseDataloader
from train.learningrate.learningrate import BaseLearningRate
from train.optimizer.optimizer import BaseOptimizer


class _registry(object):
    def __init__(self):
        self._db = {}

    def register(self, cls):
        if self._db.get(cls.__name__) is None:
            self._db[cls.__name__] = cls

    def query(self, name):  # name  cifar 10
        return self._db.get(name)


class Registry(object):
    DataLoaderRegistry = _registry()
    OptimizerRegistry = _registry()
    LRRegistry = _registry()  # lr = learningrate

    for factory_name, factory, base_cls in zip(['dataset', 'optimizer', 'learningrate'],
                                               [DataLoaderRegistry, OptimizerRegistry, LRRegistry],
                                               [BaseDataloader, BaseOptimizer, BaseLearningRate], ):
        # glob.glob  遍历该文件夹所有满足条件的文档/文件
        modules = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train', factory_name,
                                         "*.py"))  # BenchENAS/train/ dataset   learningrate optimizer
        for module in modules:
            if os.path.basename(module)[
               :-3] != '__init__':  # 不是 dataset/__init__.py      learningrate/__init__.py     optimizer/__init__.py
                md = importlib.import_module('.', 'train.' + factory_name + '.' + os.path.basename(
                    module[:-3]))  # train/dataset/cifar10_dataloader.py
                for k, v in md.__dict__.items():
                    try:
                        if issubclass(v, base_cls):  # v是base_cls的子类
                            factory.register(v)  # root   input_size  out_cla_num
                    except TypeError:
                        pass

    def __init__(self):
        pass
