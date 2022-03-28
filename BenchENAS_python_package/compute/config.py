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



