
import os
import numpy as np
from subprocess import Popen, PIPE

from algs.cnn_ga.genetic.statusupdatetool import StatusUpdateTool
from compute import Config_ini
from compute.file import get_algo_local_dir, get_local_path, get_transfer_local_path
from comm.log import Log
from algs.cnn_ga.genetic.population import Population, Individual, ResUnit, PoolUnit
import multiprocessing
import time
import platform


class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def path_replace(cls, input_str):
        new_str = input_str.replace('\\', '/')
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
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.debug('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
                file_name = cls.path_replace(file_name)
                f = open(file_name, 'a+')
                _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = '%s/begin_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        file_name = '%s/crossover_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
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
                    number_index = len(prefix) + 1
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
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(params, indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('[conv'):
                        data_maps = line[6:-1].split(',', 5)
                        conv_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                conv_params['number'] = int(_value)
                            elif _key == 'in':
                                conv_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                conv_params['out_channel'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s' % (_key))
                        conv = ResUnit(conv_params['number'], conv_params['in_channel'], conv_params['out_channel'])
                        indi.units.append(conv)
                    elif line.startswith('[pool'):
                        pool_params = {}
                        for data_item in line[6:-1].split(','):
                            _key, _value = data_item.split(':')
                            if _key == 'number':
                                indi.number_id = int(_value)
                                pool_params['number'] = int(_value)
                            elif _key == 'type':
                                pool_params['max_or_avg'] = float(_value)
                            else:
                                raise ValueError('Unknown key for load pool unit, key_name:%s' % (_key))
                        pool = PoolUnit(pool_params['number'], pool_params['max_or_avg'])
                        indi.units.append(pool)
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            pop.individuals.append(indi)
        f.close()

        # load the fitness to the individuals who have been evaluated, only suitable for the first generation
        if gen_no == 0:
            after_file_path = '%s/after_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
            if os.path.exists(after_file_path):
                fitness_map = {}
            f = open(after_file_path)
            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_map[line[0]] = float(line[1])
            f.close()

            for indi in pop.individuals:
                if indi.id in fitness_map:
                    indi.acc = fitness_map[indi.id]

        return pop

    @classmethod
    def read_template(cls):
        _path = os.path.join(os.path.dirname(__file__), 'template', 'model_template.py')
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()
        line = f.readline().rstrip()
        while line.strip() != '#generate_init':
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi, test=False):
        # query convolution unit
        conv_name_list = []
        conv_list = []
        for u in indi.units:
            if u.type == 1:
                conv_name = 'self.conv_%d_%d' % (u.in_channel, u.out_channel)
                if conv_name not in conv_name_list:
                    conv_name_list.append(conv_name)
                    conv = '%s = BasicBlock(in_planes=%d, planes=%d)' % (conv_name, u.in_channel, u.out_channel)
                    conv_list.append(conv)

        # query fully-connect layer
        out_channel_list = []
        image_output_size = StatusUpdateTool.get_input_size()
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            else:
                out_channel_list.append(out_channel_list[-1])
                image_output_size = int(image_output_size / 2)
        fully_layer_name = 'self.linear = nn.Linear(%d,%d)' % (
            image_output_size * image_output_size * out_channel_list[-1], StatusUpdateTool.get_num_class())
        # generate the forward part
        forward_list = []
        for i, u in enumerate(indi.units):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            if u.type == 1:
                _str = 'out_%d = self.conv_%d_%d(%s)' % (i, u.in_channel, u.out_channel, last_out_put)
                forward_list.append(_str)

            else:
                if u.max_or_avg < 0.5:
                    _str = 'out_%d = F.max_pool2d(%s, 2)' % (i, last_out_put)
                else:
                    _str = 'out_%d = F.avg_pool2d(%s, 2)' % (i, last_out_put)
                forward_list.append(_str)
        forward_list.append('out = out_%d' % (len(indi.units) - 1))

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        if not test:
            file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        else:
            file_name = '%s/cnn_ga_%s.py' % (os.path.join(get_transfer_local_path(), 'example'), indi.id)
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
