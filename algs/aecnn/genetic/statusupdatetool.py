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
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)
                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
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
    def get_resnet_limit(cls):
        rs = cls.__read_ini_file('network', 'resnet_limit')
        resnet_limit = []
        for i in rs.split(','):
            resnet_limit.append(int(i))
        return resnet_limit[0], resnet_limit[1]

    @classmethod
    def get_pool_limit(cls):
        rs = cls.__read_ini_file('network', 'pool_limit')
        pool_limit = []
        for i in rs.split(','):
            pool_limit.append(int(i))
        return pool_limit[0], pool_limit[1]

    @classmethod
    def get_densenet_limit(cls):
        rs = cls.__read_ini_file('network', 'densenet_limit')
        densenet_limit = []
        for i in rs.split(','):
            densenet_limit.append(int(i))
        return densenet_limit[0], densenet_limit[1]

    @classmethod
    def get_resnet_unit_length_limit(cls):
        rs = cls.__read_ini_file('resnet_configuration', 'unit_length_limit')
        resnet_unit_length_limit = []
        for i in rs.split(','):
            resnet_unit_length_limit.append(int(i))
        return resnet_unit_length_limit[0], resnet_unit_length_limit[1]

    @classmethod
    def get_densenet_k_list(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_list')
        k_list = []
        for i in rs.split(','):
            k_list.append(int(i))
        return k_list

    @classmethod
    def get_densenet_k12(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_12')
        k12_limit = []
        for i in rs.split(','):
            k12_limit.append(int(i))
        return k12_limit[0], k12_limit[1], k12_limit[2]

    @classmethod
    def get_densenet_k20(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_20')
        k20_limit = []
        for i in rs.split(','):
            k20_limit.append(int(i))
        return k20_limit[0], k20_limit[1], k20_limit[2]

    @classmethod
    def get_densenet_k40(cls):
        rs = cls.__read_ini_file('densenet_configuration', 'k_40')
        k40_limit = []
        for i in rs.split(','):
            k40_limit.append(int(i))
        return k40_limit[0], k40_limit[1], k40_limit[2]

    @classmethod
    def get_output_channel(cls):
        rs = cls.__read_ini_file('network', 'output_channel')
        channels = []
        for i in rs.split(','):
            channels.append(int(i))
        return channels

    @classmethod
    def get_input_channel(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[2]

    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num(Config_ini.dataset)

    @classmethod
    def get_input_size(cls):
        rs = TrainConfig.get_data_input_size(Config_ini.dataset)
        return rs[0]

    @classmethod
    def get_pop_size(cls):
        rs = Config_ini.pop_size
        return int(rs)

    @classmethod
    def get_max_gen(cls):
        rs = Config_ini.max_gen
        return int(rs)

    @classmethod
    def get_individual_max_length(cls):
        rs = cls.__read_ini_file('network', 'max_length')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p

    @classmethod
    def get_init_params(cls):
        params = {}
        params['max_gen'] = cls.get_max_gen()
        params['pop_size'] = cls.get_pop_size()
        params['max_len'] = cls.get_individual_max_length()
        params['image_channel'] = cls.get_input_channel()
        params['output_channel'] = cls.get_output_channel()
        params['genetic_prob'] = cls.get_genetic_probability()

        params['min_resnet'], params['max_resnet'] = cls.get_resnet_limit()
        params['min_pool'], params['max_pool'] = cls.get_pool_limit()
        params['min_densenet'], params['max_densenet'] = cls.get_densenet_limit()

        params['min_resnet_unit'], params['max_resnet_unit'] = cls.get_resnet_unit_length_limit()

        params['k_list'] = cls.get_densenet_k_list()
        params['max_k12_input_channel'], params['min_k12'], params['max_k12'] = cls.get_densenet_k12()
        params['max_k20_input_channel'], params['min_k20'], params['max_k20'] = cls.get_densenet_k20()
        params['max_k40_input_channel'], params['min_k40'], params['max_k40'] = cls.get_densenet_k40()

        return params

    @classmethod
    def get_mutation_probs_for_each(cls):
        """
        defined the particular probabilities for each type of mutation
        the mutation occurs at:
        --    add
        -- remove
        --  alter
        """
        rs = cls.__read_ini_file('settings', 'mutation_probs').split(',')
        assert len(rs) == 3
        mutation_prob_list = [float(i) for i in rs]
        return mutation_prob_list
