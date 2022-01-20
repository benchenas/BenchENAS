import numpy as np
import torch

from comm.registry import Registry
from compute import Config_ini
from tools import StatusUpdateTool
from train.dataset.dataloader import BaseDataloader


def test_cifar_loader():
    Config_ini.batch_size = 64
    Config_ini.total_epoch = 50
    datasets = ['CIFAR10', 'CIFAR100']
    for dataset in datasets:
        dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
        dataloader_cls_ins = dataloader_cls()
        dataloader_cls_ins.amend_valid_size(val=0.2)
        train_dataloader = dataloader_cls_ins.get_train_dataloader()
        valid_loader = dataloader_cls_ins.get_val_dataloader()
        assert len(train_dataloader) == np.ceil((1 - dataloader_cls_ins.valid_size) * 50000 / 64)
        assert len(valid_loader) == np.ceil(dataloader_cls_ins.valid_size * 50000 / 64)

    for dataset in datasets:
        dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
        dataloader_cls_ins = dataloader_cls()
        dataloader_cls_ins.amend_valid_size(val=None)
        train_dataloader = dataloader_cls_ins.get_train_dataloader()
        valid_loader = dataloader_cls_ins.get_val_dataloader()
        assert len(train_dataloader) == np.ceil(50000 / 64)
        assert len(valid_loader) == np.ceil(10000 / 64)


def test_end_evolution():
    algs = ['aecnn', 'cgp_cnn', 'cnn_ga', 'genetic_CNN', 'hierarchical_representations',
            'large_scale', 'regularized_evolution', 'nsga_net']
    section = 'evolution_status'
    key = 'IS_RUNNING'
    for alg in algs:
        tool = StatusUpdateTool(alg)
        tool.end_evolution()
        val = tool.read_ini_file(section, key)
        assert int(val) == 0
