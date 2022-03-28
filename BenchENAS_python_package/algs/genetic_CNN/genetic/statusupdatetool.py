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
    def get_num_nodes(cls):
        rs = cls.__read_ini_file('settings', 'NUM_NODES')
        num_nodes = []
        for i in rs.split(','):
            num_nodes.append(int(i))
        return num_nodes

    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num(Config_ini.dataset)

    @classmethod
    def get_input_size(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[0]

    @classmethod
    def get_input_channel(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[2]

    @classmethod
    def get_stages(cls):
        rs = cls.__read_ini_file('settings', 'STAGES')
        stages = []
        for i in rs.split(','):
            stages.append(i)
        return stages

    @classmethod
    def get_mutation_prob(cls):
        rs = cls.__read_ini_file('settings', 'mutation_prob')
        return float(rs)

    @classmethod
    def get_crossover_prob(cls):
        rs = cls.__read_ini_file('settings', 'crossover_prob')
        return float(rs)

    @classmethod
    def get_init_params(cls):
        pop_size = Config_ini.pop_size
        max_gen = Config_ini.max_gen
        params = {}
        params['pop_size'] = pop_size
        params['max_gen'] = max_gen
        params['NUM_NODES'] = cls.get_num_nodes()
        params['STAGES'] = cls.get_stages()
        params['mutation_prob'] = cls.get_mutation_prob()
        params['crossover_prob'] = cls.get_crossover_prob()
        l, _, _ = cls.get_params()
        params['l'] = l
        return params

    @classmethod
    def get_params(cls):
        L = 0
        BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0  # to keep track of bits for each stage S
        for nn in cls.get_num_nodes():
            t = nn * (nn - 1)
            BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
            l_bpi += int(0.5 * t)
            L += t
        L = int(0.5 * L)
        return L, BITS_INDICES, l_bpi

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
