import os
import configparser


class Config(object):

    def __init__(self, config_file, section):
        self.config_file = config_file
        self.section = section

    def write_ini_file(self, key, value):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        config.set(self.section, key, value)
        config.write(open(self.config_file, 'w'))

    def read_ini_file(self, key):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config.get(self.section, key)


class AlgorithmConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global.ini')  # BenchENAS/global.ini
        Config.__init__(self, file_path, 'algorithm')


class RedisConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global.ini')
        Config.__init__(self, file_path, 'redis')


class ExecuteConfig(Config):
    def __init__(self):
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global.ini')
        Config.__init__(self, file_path, 'execute')


if __name__ == '__main__':
    g = AlgorithmConfig()
    # db_ip = g.read_ini_file('log_server')
    db_port = int(g.read_ini_file('pop_size'))
    print(db_port + 1)
