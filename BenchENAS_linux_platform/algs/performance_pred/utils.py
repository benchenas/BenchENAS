import configparser
import os
import logging
import sys

from algs.performance_pred.units.network import CNN
from compute.file import get_population_dir
from train.utils import TrainConfig


class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__),'global.ini'))
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)
                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
        config.write(open(os.path.join(os.path.dirname(__file__),'global.ini'), 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.join(os.path.dirname(__file__),'global.ini')))
        config.set(section, key, value)
        config.write(open(os.path.join(os.path.dirname(__file__),'global.ini'), 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__),'global.ini'))
        return config.get(section, key)
    
    @classmethod
    def get_output_class_num(cls):
        num = cls.__read_ini_file('global', 'out_class_num')
        return int(num)

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
    def _read_network_parameters_helper(cls, section, key, value_type):
        
        if value_type not in ['int', 'float']:
            
            exit('%s is not a valid type'%(value_type))
            
        
        rs = cls.__read_ini_file(section, key)
        if rs.startswith('['):
            rs_1 = rs[1:-1].split(',')
            rs_2 = [i.strip() for i in rs_1]
            if value_type == 'int':
                rs_3 = [int(i) for i in rs_2]
            else:
                rs_3 = [float(i) for i in rs_2]
            return rs_3
            
        else:
            if value_type == 'int':
                return int(rs)
            else:
                return float(rs)
        
    
    @classmethod
    def get_sample_number(cls):
        return cls._read_network_parameters_helper('global', 'samples', 'int')

    @classmethod
    def get_network_params(cls):
        params = {}
        # params['input_size'] = cls._read_network_parameters_helper('network', 'input_size', 'int')
        params['input_size'] = TrainConfig.get_data_input_size()
        params['max_length'] = cls._read_network_parameters_helper('network', 'max_length', 'int')
        params['min_length'] = cls._read_network_parameters_helper('network', 'min_length', 'int')
        params['pool_amount'] = cls._read_network_parameters_helper('network', 'pool_amount', 'int')
        params['pool_amount_prob'] = cls._read_network_parameters_helper('network', 'pool_amount_prob', 'float')
        params['pool_type_prob'] = cls._read_network_parameters_helper('network', 'pool_type_prob', 'float')
        params['full_amount'] = cls._read_network_parameters_helper('network', 'full_amount', 'int')
        params['full_amount_prob'] = cls._read_network_parameters_helper('network', 'full_amount_prob', 'float')
        params['conv_output_channel'] = cls._read_network_parameters_helper('network', 'conv_output_channel', 'int')
        params['conv_output_channel_prob'] = cls._read_network_parameters_helper('network', 'conv_output_channel_prob', 'float')
        params['conv_kernel'] = cls._read_network_parameters_helper('network', 'conv_kernel', 'int')
        params['conv_kernel_prob'] = cls._read_network_parameters_helper('network', 'conv_kernel_prob', 'float')
        params['conv_stride'] = cls._read_network_parameters_helper('network', 'conv_stride', 'int')
        params['conv_stride_prob'] = cls._read_network_parameters_helper('network', 'conv_stride_prob', 'float')
        params['conv_groups'] = cls._read_network_parameters_helper('network', 'conv_groups', 'int')
        params['conv_groups_prob'] = cls._read_network_parameters_helper('network', 'conv_groups_prob', 'float')
        params['conv_padding'] = cls._read_network_parameters_helper('network', 'conv_padding', 'int')
        params['conv_padding_prob'] = cls._read_network_parameters_helper('network', 'conv_padding_prob', 'float')
        params['conv_add_to_prob'] = cls._read_network_parameters_helper('network', 'conv_add_to_prob', 'float')
        params['conv_concatenate_to_prob'] = cls._read_network_parameters_helper('network', 'conv_concatenate_to_prob', 'float')
        params['pool_kernel'] = cls._read_network_parameters_helper('network', 'pool_kernel', 'int')
        params['pool_kernel_prob'] = cls._read_network_parameters_helper('network', 'pool_kernel_prob', 'float')
        params['pool_stride'] = cls._read_network_parameters_helper('network', 'pool_stride', 'int')
        params['pool_stride_prob'] = cls._read_network_parameters_helper('network', 'pool_stride_prob', 'float')
        params['pool_padding'] = cls._read_network_parameters_helper('network', 'pool_padding', 'int')
        params['pool_padding_prob'] = cls._read_network_parameters_helper('network', 'pool_padding_prob', 'float')
        params['full_size'] = cls._read_network_parameters_helper('network', 'full_size', 'int')
        params['full_size_prob'] = cls._read_network_parameters_helper('network', 'full_size_prob', 'float')
        params['full_dropout'] = cls._read_network_parameters_helper('network', 'full_dropout', 'float')
        params['full_dropout_prob'] = cls._read_network_parameters_helper('network', 'full_dropout_prob', 'float')

        return params

class Log(object):
    _logger = None
    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("SampleCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)
    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)

                

class GenerateCNNToFile():
    def __init__(self, number_to_generate):
        
        self.number_to_generate = number_to_generate
        
    def save_to_file(self, no, cnn):
        pop_dir = get_population_dir()

        if no == 0:
            f = open(os.path.join(pop_dir, 'networks.txt'), 'w')
        else:
            f = open(os.path.join(pop_dir, 'networks.txt'), 'a+')
        f.write(str(cnn))
        f.write('\n')
        f.write('='*200)
        f.write('\n')
        f.flush()
        f.close()
    
    
    def do(self):
        params = StatusUpdateTool.get_network_params()
        for i in range(self.number_to_generate):
            cnn = CNN(i, params)
            cnn.create()
            self.save_to_file(i, cnn)
            Log.info('Generate CNN and save to file, %d/%d finished'%(i+1, self.number_to_generate))
            

if __name__ == '__main__':
    params = StatusUpdateTool.get_network_params()
    print(StatusUpdateTool.get_epoch_size())
    
    