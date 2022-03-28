import configparser
import os
import platform
import multiprocessing
from compute.file import get_algo_local_dir
import time
import collections
# from compute.config import AlgorithmConfig
# import numpy as np
from algs.regularized_evolution.genetic.population import Population, Individual
from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
Genotype = collections.namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

class Utils(object):
    _lock = multiprocessing.Lock()
    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def path_replace(cls, input_str):
        # input a str, replace '\\' with '/', because the os.path in windows return path with '\\' joining
        # please use it after creating a string with both os.path and string '/'
        if (platform.system() == 'Windows'):
            new_str = input_str.replace('\\', '/')
        else:  # Linux or Mac
            new_str = input_str
        return new_str

    @classmethod
    def load_cache_data(cls):
        file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
        file_name = cls.path_replace(file_name)
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f' % (float(rs_[1]))
            f.close()
        return _map


    @classmethod
    def save_population_at_begin(cls, _str):
        file_name = '%s/begin.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
        # solve the path differences caused by different platforms
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)


    @classmethod
    def save_file_at_name(cls, _str, name):
        file_name = '%s/%s.txt' % (os.path.join(get_algo_local_dir(), 'populations'), name)
        # solve the path differences caused by different platforms
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)


    @classmethod
    def load_population(cls, name):
        file_name = '%s\%s.txt' % (os.path.join(get_algo_local_dir(), 'populations'), name)
        file_name = cls.path_replace(file_name)
        params = StatusUpdateTool.get_init_params()
        pop = Population(params)
        if os.path.exists(file_name) == False:
            return None
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        acc = float(line[4:])
                    elif line.startswith('normal_cell:'):
                        normal_cell = eval(line[12:])
                    elif line.startswith('reduction_cell'):
                        reduction_cell = eval(line[15:])
                    elif line.startswith('genotype:'):
                        genotype = eval(line[9:])
            indi = Individual(indi_no, params, tuple=(normal_cell, reduction_cell))
            indi.acc = acc
            pop.individuals.append(indi)
        f.close()
        return pop

    @classmethod
    def read_template(cls):
        _path = os.path.join(os.path.dirname(__file__), 'template', 'model_template.py')
        part1 = []
        part2 = []
        f = open(_path)
        f.readline()  # skip this comment

        line = f.readline().rstrip()
        while line.strip() != "# geno":
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part2.append(line)
            line = f.readline().rstrip()
        return part1, part2

    @classmethod
    def generate_pytorch_file(cls, indi, params):
        geno = indi.genotype
        part1, part2 = cls.read_template()
        s1 = 'genotype = ' + str(geno)
        s2 = 'self.net = NetworkCIFAR('+str(params['F'])+' ,'+str(StatusUpdateTool.get_num_class())+', '+str(params['N'])+', True, genotype)'

        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('        %s' % (s1))
        _str.append('        %s' % (s2))

        _str.extend(part2)
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        file_name = cls.path_replace(file_name)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()


    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()
