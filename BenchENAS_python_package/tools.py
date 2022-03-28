import configparser
import os


class StatusUpdateTool(object):
    def __init__(self, alg_name):
        if alg_name != 'cnn_ga' and alg_name != 'nsga_net':
            self.config_file = os.path.join(os.path.dirname(__file__), 'algs', alg_name, 'genetic', 'global.ini')
        elif alg_name == 'cnn_ga':
            self.config_file = os.path.join(os.path.dirname(__file__), 'algs', alg_name, 'global.ini')
        else:
            self.config_file = os.path.join(os.path.dirname(__file__), 'algs', alg_name, 'utils', 'global.ini')

    def write_ini_file(self, section, key, value):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        config.set(section, key, value)
        config.write(open(self.config_file, 'w'))

    def read_ini_file(self, section, key):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config.get(section, key)

    def end_evolution(self):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        self.write_ini_file(section, key, "0")