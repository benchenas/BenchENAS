# coding=utf-8

import os, glob, importlib
from train.optimizer.optimizer import BaseOptimizer
from train.dataset.dataloader import BaseDataloader
from train.learningrate.learningrate import BaseLearningRate


class _registry():
    def __init__(self):
        self._db = {}

    def register(self, cls):
        if self._db.get(cls.__name__) is None:
            self._db[cls.__name__] = cls

    def query(self, name):
        return self._db.get(name)


class Registry():
    DataLoaderRegistry = _registry()
    OptimizerRegistry = _registry()
    LRRegistry = _registry()

    for factory_name, factory, base_cls in zip(['dataset', 'optimizer', 'learningrate'],
                                               [DataLoaderRegistry, OptimizerRegistry, LRRegistry],
                                               [BaseDataloader, BaseOptimizer, BaseLearningRate], ):

        modules = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train', factory_name, "*.py"))
        for module in modules:
            if os.path.basename(module)[:-3] != '__init__':
                md = importlib.import_module('.', 'train.' + factory_name + '.' + os.path.basename(module[:-3]))
                for k, v in md.__dict__.items():
                    try:
                        if issubclass(v, base_cls):
                            factory.register(v)
                    except TypeError:
                        pass

    def __init__(self):
        pass
