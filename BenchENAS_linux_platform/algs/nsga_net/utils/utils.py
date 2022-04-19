import configparser
import os
import platform
import multiprocessing
from compute.file import get_algo_local_dir
import time
import os
import numpy as np
from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
from algs.nsga_net.genetic.population import Population, Individual


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
    def save_fitness_to_cache(cls, individuals):
        _map1, _map2 = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            _flop = indi.flop
            if _key not in _map:
                file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
                file_name = cls.path_replace(file_name)
                f = open(file_name, 'a+')
                _str = '%s;%.5f;%.5f;%s\n' % (_key, _acc, _flop, _str)
                f.write(_str)
                f.close()
                _map1[_key] = _acc
                _map2[_key] = _flop

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = '%s/begin_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        # solve the path differences caused by different platforms
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = '%s/mutation_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk(os.path.join(get_algo_local_dir(), 'populations')):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    number_index = len(prefix) + 1  # the first number index
                    id_list.append(int(file_name[number_index:number_index + 5]))
        if len(id_list) == 0:

            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = '%s/%s_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), prefix, np.min(gen_no))
        file_name = cls.path_replace(file_name)
        params = StatusUpdateTool.get_init_params()
        pop = Population(gen_no, params)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(indi_no, params, params['n_var'])
            genome = []
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('flop'):
                        indi.flop = float(line[5:])
                    elif line.startswith('genome'):
                        print(line)
                        l = list(line[8:])
                        while ' ' in l:
                            l.remove(' ')
                        while ',' in l:
                            l.remove(',')
                        while ']' in l:
                            l.remove(']')
                        for i in l:
                            genome.append(int(i))
                    elif line.startswith('0') or line.startswith('1'):
                        print(line)
                        l = list(line)
                        while ' ' in l:
                            l.remove(' ')
                        while ',' in l:
                            l.remove(',')
                        while ']' in l:
                            l.remove(']')
                        for i in l:
                            genome.append(int(i))
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            indi.genome = np.array(genome)
            pop.individuals.append(indi)
        f.close()
        return pop

    @classmethod
    def read_template(cls, search_space):
        _path = os.path.join(os.path.dirname(__file__), 'template', search_space + '_models.py')
        part1 = []
        part2 = []
        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != "#generate_init":
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part2.append(line)
            line = f.readline().rstrip()
        return part1, part2

    @classmethod
    def generate_micro_pytorch_file(cls, indi, params):
        search_space = "micro"
        part1, part2 = cls.read_template(search_space)
        line1 = "genome = convert(%s)" % (str(list(indi.genome)))
        line2 = "genotype = decode(genome)"
        line3 = "self.net = Network(%d, %d, %d, %d, False, genotype)" % \
                (StatusUpdateTool.get_input_channel(), params['init_channels'], params['classes'], params['layers'])
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('        %s' % (line1))
        _str.append('        %s' % (line2))
        _str.append('        %s' % (line3))
        _str.extend(part2)
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        file_name = cls.path_replace(file_name)
        if not os.path.exists(os.path.join(get_algo_local_dir(), 'scripts')):
            os.makedirs(os.path.join(get_algo_local_dir(), 'scripts'))
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def generate_macro_pytorch_file(cls, indi, channels, params):
        search_space = "macro"
        part1, part2 = cls.read_template(search_space)
        line1 = "genome = convert(np.array(%s))" % (str(list(indi.genome)))
        line2 = "genotype = decode(genome)"
        line3 = "channels = %s" % (str(channels))
        line4 = "self.net = EvoNetwork(genotype, channels, %d, (32, 32), decoder='residual')" % \
                (params['classes'])
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('        %s' % (line1))
        _str.append('        %s' % (line2))
        _str.append('        %s' % (line3))
        _str.append('        %s' % (line4))
        _str.extend(part2)
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        file_name = cls.path_replace(file_name)
        if not os.path.exists(os.path.join(get_algo_local_dir(), 'scripts')):
            os.makedirs(os.path.join(get_algo_local_dir(), 'scripts'))
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
