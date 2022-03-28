import configparser
import os
from compute.config import AlgorithmConfig
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
    def get_imgSize(cls):
        rs = TrainConfig.get_data_input_size()
        return int(rs[0])

    @classmethod
    def get_input_channel(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[2]

    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num()

    @classmethod
    def get_lam(cls):
        rs = cls.__read_ini_file('settings', 'lam')
        return int(rs)

    @classmethod
    def get_rows(cls):
        rs = cls.__read_ini_file('settings', 'rows')
        return int(rs)

    @classmethod
    def get_cols(cls):
        rs = cls.__read_ini_file('settings', 'cols')
        return int(rs)

    @classmethod
    def get_level_back(cls):
        rs = cls.__read_ini_file('settings', 'level_back')
        return int(rs)

    @classmethod
    def get_min_active_num(cls):
        rs = cls.__read_ini_file('settings', 'min_active_num')
        return int(rs)

    @classmethod
    def get_max_active_num(cls):
        rs = cls.__read_ini_file('settings', 'max_active_num')
        return int(rs)

    @classmethod
    def get_mutation_rate(cls):
        rs = cls.__read_ini_file('settings', 'mutation_rate')
        return float(rs)

    @classmethod
    def get_init_params(cls):
        g = AlgorithmConfig()
        pop_size = int(g.read_ini_file('pop_size'))
        max_gen = int(g.read_ini_file('max_gen'))
        params = {}
        params['pop_size'] = pop_size
        params['max_gen'] = max_gen
        params['imgSize'] = cls.get_imgSize()
        params['lam'] = cls.get_lam()
        params['rows'] = cls.get_rows()
        params['cols'] = cls.get_cols()
        params['level_back'] = cls.get_level_back()
        params['min_active_num'] = cls.get_min_active_num()
        params['max_active_num'] = cls.get_max_active_num()
        params['mutation_rate'] = cls.get_mutation_rate()
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