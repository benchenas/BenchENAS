import os
import sys
import numpy as np

from compute import Config_ini
from train.dataset.comm_data import FDataLoader


def test_customized_loader():
    Config_ini.batch_size = 50
    Config_ini.total_epoch = 50
    os.chdir(sys.path[0])
    data_dir = os.path.join(os.getcwd(), 'example/eye_dataset')
    print('dir', data_dir)
    Config_ini.dataset = 'customized'
    Config_ini.data_dir = data_dir
    Config_ini.img_input_size = [244, 244, 3]

    dataloader_cls_ins = FDataLoader()
    assert dataloader_cls_ins.train_test == 1
    train_loader = dataloader_cls_ins.get_train_dataloader()
    valid_loader = dataloader_cls_ins.get_val_dataloader()

    assert len(train_loader) == np.ceil(100 / 50)
    assert len(valid_loader) == np.ceil(100 / 50)

    dataloader_cls_ins.train_test = 2
    dataloader_cls_ins.amend_valid_size(val=0.3)
    train_dataloader = dataloader_cls_ins.get_train_dataloader()
    valid_loader = dataloader_cls_ins.get_val_dataloader()
    assert len(train_dataloader) == np.ceil((1 - dataloader_cls_ins.valid_size) * 100 / 50)
    assert len(valid_loader) == np.ceil(dataloader_cls_ins.valid_size * 100 / 50)
