import os, configparser
from compute.file import get_algo_local_dir
from compute.redis import RedisLog


class PlatENASConfig(object):

    def __init__(self, section):
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global.ini')
        self.config_file = file_path
        self.section = section

    def read_ini_file(self, key):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        return config.get(self.section, key)


class GPUFitness(object):
    @classmethod
    def read(cls):
        file_name = '%s/populations/results.txt' % (get_algo_local_dir())
        if not os.path.exists(file_name):
            file = open(file_name, 'w')
            file.close()
        f = open(file_name, 'r')
        fitness_map = {}
        for line in f:
            if len(line.strip()) > 0:
                line = line.strip().split('=')
                fitness_map[line[0]] = float(line[1])
        f.close()
        return fitness_map


class CacheToResultFile(object):
    @classmethod
    def do(cls, file_id, best_acc, db_ip, db_port):
        logger = RedisLog(os.path.basename(file_id) + '.txt', db_ip, db_port)
        logger.write_file('RESULTS', 'results.txt', '%s=%.5f\n' % (file_id, best_acc))



