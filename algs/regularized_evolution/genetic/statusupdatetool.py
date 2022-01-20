import configparser
import os
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
    def get_N(cls):
        rs = cls.__read_ini_file('settings', 'N')
        return int(rs)

    @classmethod
    def get_F(cls):
        rs = cls.__read_ini_file('settings', 'F')
        return int(rs)

    @classmethod
    def get_sample_size(cls):
        rs = cls.__read_ini_file('settings', 'sample_size')
        return int(rs)

    @classmethod
    def get_cycles(cls):
        rs = cls.__read_ini_file('settings', 'cycles')
        return int(rs)

    @classmethod
    def get_conv_size(cls):
        rs = cls.__read_ini_file('settings', 'conv_size')
        conv_size = []
        for i in rs.split(','):
            conv_size.append(int(i))
        return conv_size

    @classmethod
    def get_init_params(cls):
        pop_size = Config_ini.pop_size
        max_gen = Config_ini.max_gen
        params = {}
        params['pop_size'] = pop_size
        params['max_gen'] = max_gen
        params['N'] = cls.get_N()
        params['F'] = cls.get_F()
        params['sample_size'] = cls.get_sample_size()
        params['conv_size'] = cls.get_conv_size()
        params['cycles'] = cls.get_cycles()
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
    def change_cycles(cls, value):
        cls.__write_ini_file('settings', 'cycles', str(value))

    @classmethod
    def change_sample_size(cls, value):
        cls.__write_ini_file('settings', 'sample_size', str(value))

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False

    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num(Config_ini.dataset)
