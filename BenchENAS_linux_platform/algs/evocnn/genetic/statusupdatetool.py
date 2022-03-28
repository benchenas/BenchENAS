import configparser
import os
from train.utils import TrainConfig
from comm.utils import PlatENASConfig


class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config_file = os.path.join(os.path.dirname(__file__) ,'global.ini')
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
        config_file = os.path.join(os.path.dirname(__file__) ,'global.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        config.set(section, key, value)
        config.write(open(config_file, 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config_file = os.path.join(os.path.dirname(__file__) ,'global.ini')
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
    def get_conv_limit(cls):
        rs = cls.__read_ini_file('network','conv_num_limit')
        conv_limit = []
        for i in rs.split(','):
            conv_limit.append(int(i))
        return conv_limit[0], conv_limit[1]

    @classmethod
    def get_pool_limit(cls):
        rs = cls.__read_ini_file('network','pool_num_limit')
        pool_limit = []
        for i in rs.split(','):
            pool_limit.append(int(i))
        return pool_limit[0], pool_limit[1]

    @classmethod
    def get_fc_limit(cls):
        rs = cls.__read_ini_file('network', 'fc_num_limit')
        fc_limit = []
        for i in rs.split(','):
            fc_limit.append(int(i))
        return fc_limit[0], fc_limit[1]

    @classmethod
    def get_conv_filter_size_limit(cls):
        rs = cls.__read_ini_file('convolution_configuration','filter_size_limit')
        conv_filter_size_limit = []
        for i in rs.split(','):
            conv_filter_size_limit.append(int(i))
        return conv_filter_size_limit[0], conv_filter_size_limit[1]

    @classmethod
    def get_channel_limit(cls):
        rs = cls.__read_ini_file('convolution_configuration','channel_limit')
        channel_limit = []
        for i in rs.split(','):
            channel_limit.append(int(i))
        return channel_limit[0], channel_limit[1]

    @classmethod
    def get_pool_kernel_size_list(cls):
        rs = cls.__read_ini_file('pool_configuration','kernel_size_list')
        pool_filter_list = []
        for i in rs.split(','):
            pool_filter_list.append(int(i))
        return pool_filter_list

    @classmethod
    def get_hidden_neurons_limit(cls):
        rs = cls.__read_ini_file('fc_configuration','hidden_neurons_limit')
        hidden_neurons_limit = []
        for i in rs.split(','):
            hidden_neurons_limit.append(int(i))
        return hidden_neurons_limit[0],hidden_neurons_limit[1]

    @classmethod
    def get_std_limit(cls):
        rs = cls.__read_ini_file('network','std_limit')
        std_limit = []
        for i in rs.split(','):
            std_limit.append(float(i))
        return std_limit[0], std_limit[1]

    @classmethod
    def get_mean_limit(cls):
        rs = cls.__read_ini_file('network','mean_limit')
        mean_limit = []
        for i in rs.split(','):
            mean_limit.append(float(i))
        return mean_limit[0],mean_limit[1]

    @classmethod
    def get_crossover_eta(cls):
        rs = cls.__read_ini_file('settings','crossover_eta')
        return float(rs)

    @classmethod
    def get_mutation_eta(cls):
        rs = cls.__read_ini_file('settings','mutation_eta')
        return float(rs)

    @classmethod
    def get_acc_mean_threshold(cls):
        rs = cls.__read_ini_file('settings', 'acc_mean_threshold')
        return float(rs)

    @classmethod
    def get_complexity_threshold(cls):
        rs = cls.__read_ini_file('settings', 'complexity_threshold')
        return int(rs)

    @classmethod
    def get_input_channel(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[2]
    @classmethod
    def get_num_class(cls):
        return TrainConfig.get_out_cls_num()
    @classmethod
    def get_input_size(cls):
        rs = TrainConfig.get_data_input_size()
        return rs[0]

    @classmethod
    def get_pop_size(cls):
        g = PlatENASConfig('algorithm')
        rs = g.read_ini_file('pop_size')
        return int(rs)

    @classmethod
    def get_max_gen(cls):
        g = PlatENASConfig('algorithm')
        rs = g.read_ini_file('max_gen')
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
        params['image_channel'] = cls.get_input_channel()
        params['genetic_prob'] = cls.get_genetic_probability()

        params['min_conv'], params['max_conv'] = cls.get_conv_limit()
        params['min_pool'], params['max_pool'] = cls.get_pool_limit()
        params['min_fc'], params['max_fc'] = cls.get_fc_limit()
        params['min_std'], params['max_std'] = cls.get_std_limit()
        params['min_mean'], params['max_mean'] = cls.get_mean_limit()

        params['conv_filter_size_min'], params['conv_filter_size_max'] = cls.get_conv_filter_size_limit()
        params['min_channel'], params['max_channel'] = cls.get_channel_limit()
        params['pool_kernel_size_list'] = cls.get_pool_kernel_size_list()
        params['min_hidden_neurons'], params['max_hidden_neurons'] = cls.get_hidden_neurons_limit()

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