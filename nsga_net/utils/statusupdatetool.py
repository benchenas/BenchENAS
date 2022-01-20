import configparser
import os
import numpy as np

from compute import Config_ini
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
    def get_search_space(cls):
        rs = cls.__read_ini_file('settings', 'search_space')
        return str(rs)

    @classmethod
    def get_n_blocks(cls):
        rs = cls.__read_ini_file('settings', 'n_blocks')
        return int(rs)

    @classmethod
    def get_n_ops(cls):
        rs = cls.__read_ini_file('settings', 'n_ops')
        return int(rs)

    @classmethod
    def get_n_cells(cls):
        rs = cls.__read_ini_file('settings', 'n_cells')
        return int(rs)

    @classmethod
    def get_n_nodes(cls):
        rs = cls.__read_ini_file('settings', 'n_nodes')
        return int(rs)

    @classmethod
    def get_n_offspring(cls):
        rs = cls.__read_ini_file('settings', 'n_offspring')
        return int(rs)

    @classmethod
    def get_classes(cls):
        rs = TrainConfig.get_out_cls_num(Config_ini.dataset)
        return int(rs)

    @classmethod
    def get_input_weight(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[0]

    @classmethod
    def get_input_height(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[1]

    @classmethod
    def get_init_channels(cls):
        rs = cls.__read_ini_file('settings', 'init_channels')
        return int(rs)

    @classmethod
    def get_layers(cls):
        rs = cls.__read_ini_file('settings', 'layers')
        return int(rs)

    @classmethod
    def get_init_params(cls):
        pop_size = Config_ini.pop_size
        max_gen = Config_ini.max_gen
        params = {}
        params['pop_size'] = pop_size
        params['max_gen'] = max_gen
        params['search_space'] = cls.get_search_space()
        params['n_blocks'] = cls.get_n_blocks()
        params['n_ops'] = cls.get_n_ops()
        params['n_cells'] = cls.get_n_cells()
        params['n_nodes'] = cls.get_n_nodes()
        params['n_offspring'] = pop_size
        params['n_offsprings'] = pop_size
        params['classes'] = cls.get_classes()
        params['layers'] = cls.get_layers()
        params['n_survive'] = pop_size
        params['init_channels'] = cls.get_init_channels()
        if params['search_space'] == 'micro':  # NASNet search space
            params['n_var'] = int(4 * params['n_blocks'] * 2)
            lb = np.zeros(params['n_var'])
            ub = np.ones(params['n_var'])
            h = 1
            for b in range(0, params['n_var'] // 2, 4):
                ub[b] = params['n_ops'] - 1
                ub[b + 1] = h
                ub[b + 2] = params['n_ops'] - 1
                ub[b + 3] = h
                h += 1
            ub[params['n_var'] // 2:] = ub[:params['n_var'] // 2]
            params['ub'] = ub
            params['lb'] = lb
        elif params['search_space'] == 'macro':  # modified GeneticCNN search space
            params['n_var'] = int(((params['n_nodes'] - 1) * params['n_nodes'] / 2 + 1) * 3)
            params['lb'] = np.zeros(params['n_var'])
            params['ub'] = np.ones(params['n_var'])
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

    @classmethod
    def change_search_space(cls, search_space):
        section = 'settings'
        key = 'search_space'
        cls.__write_ini_file(section, key, search_space)