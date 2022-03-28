import configparser
import os
import platform
import multiprocessing
from compute.file import get_algo_local_dir
import time
from compute.config import AlgorithmConfig
import numpy as np

from train.utils import TrainConfig


class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config_file = os.path.join(os.path.dirname(__file__), 'global.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        config.write(open(config_file, 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config_file = os.path.join(os.path.dirname(__file__), 'global.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        config.set(section, key, value)
        config.write(open(config_file, 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config_file = os.path.join(os.path.dirname(__file__), 'global.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        return config.get(section, key)

    @classmethod
    def get_Gs(cls):
        rs = cls.__read_ini_file('settings', 'Gs')
        num_nodes = []
        for i in rs.split(','):
            num_nodes.append(int(i))
        return num_nodes

    @classmethod
    def get_Ms(cls):
        rs = cls.__read_ini_file('settings', 'Ms')
        num_nodes = []
        for i in rs.split(','):
            num_nodes.append(int(i))
        return num_nodes

    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num()

    @classmethod
    def get_input_weight(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[0]

    @classmethod
    def get_input_height(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[1]

    @classmethod
    def get_input_channel(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[2]

    @classmethod
    def get_L(cls):
        rs = cls.__read_ini_file('settings', 'L')
        return int(rs)

    @classmethod
    def get_init_params(cls):
        g = AlgorithmConfig()
        pop_size = int(g.read_ini_file('pop_size'))
        max_gen = int(g.read_ini_file('max_gen'))
        params = {}
        params['pop_size'] = pop_size
        params['max_gen'] = max_gen
        params['L'] = cls.get_L()
        params['Ms'] = cls.get_Ms()
        params['Gs'] = cls.get_Gs()
        return params

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False