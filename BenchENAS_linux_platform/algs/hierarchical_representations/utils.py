import configparser
import os
import platform
import multiprocessing
from compute.file import get_algo_local_dir
import time
import math
from compute.config import AlgorithmConfig
import numpy as np
from algs.hierarchical_representations.genetic.population import Population, Individual
from algs.hierarchical_representations.genetic.statusupdatetool import StatusUpdateTool

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
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
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
            indi = Individual(indi_no, params)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('['):
                        l = list(line[1:-1])
                        for i in range(0, len(l)):
                            l[i] = int(l[i])
                        indi.indi = l
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            pop.individuals.append(indi)
        f.close()
        return pop

    @classmethod
    def read_template(cls):
        _path = os.path.join(os.path.dirname(__file__), 'template', 'model_template.py')
        part1 = []
        part2 = []
        part3 = []
        part4 = []
        part5 = []
        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_motif':
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '#generate_conv':
            part3.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '#linear_layer':
            part4.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '"""':
            part5.append(line)
            line = f.readline().rstrip()
        return part1, part2, part3, part4, part5

    @classmethod
    def generate_all(cls, indi, motif_str):
        part1, part2, part3, part4, part5 = cls.read_template()
        conv_unit = '        self.conv3x3 = nn.Conv2d(%d, 16, kernel_size=3, stride=1, padding=1)' \
                    % (StatusUpdateTool.get_input_channel())
        linear_layer = '        self.last_linear = nn.Linear(%d, %d)' % \
                       (64 * math.pow(StatusUpdateTool.get_input_height() / 4, 2), StatusUpdateTool.get_num_class())
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        for i in motif_str:
            _str.extend(i)
        _str.extend(part2)
        _str.append('       return self.layer1(input)')
        _str.extend(part3)
        _str.append(conv_unit)
        _str.extend(part4)
        _str.append(linear_layer)
        _str.extend(part5)
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        file_name = cls.path_replace(file_name)
        if not os.path.exists(os.path.join(get_algo_local_dir(), 'scripts')):
            os.makedirs(os.path.join(get_algo_local_dir(), 'scripts'))
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()



    @classmethod
    def generate_layer(cls, net):
        layer_list = []
        layer_num = 1

        # generate the forward part
        forward_list = []
        final_list = []

        for i, u in enumerate(net.without_predecessors):
            _str = '%s = input' % (u)
            forward_list.append(_str)

        for i, u in enumerate(net.units):
            layer_list.append("self.layer%d = %s(in_channel)" % (layer_num, u.type))
            _str = '%s = self.layer%d(%s)' % (u.out_name, layer_num, u.in_name)
            layer_num = layer_num + 1
            forward_list.append(_str)

            for j, (ind_node_name, dep_node_name) in enumerate(net.skipconnections):
                if dep_node_name == u.out_name:
                    _str = 'residual_%s = %s' % (ind_node_name, ind_node_name)
                    forward_list.append(_str)
                    _str = '%s = %s + residual_%s' % (dep_node_name, dep_node_name, ind_node_name)
                    forward_list.append(_str)

            for j, node_name in enumerate(net.without_towards):
                if node_name == u.out_name:
                    if len(final_list) == 0:
                        _str = 'final = %s' % (node_name)
                        forward_list.append(_str)
                        final_list.append(node_name)
                    elif len(final_list) > 0:
                        _str = 'final = final + %s' % (node_name)
                        forward_list.append(_str)
                        final_list.append(node_name)

        _str = []
        _str.append('class %s(nn.Module):' % (net.motif_name))
        _str.append('    def __init__(self, in_channel):')
        _str.append('        super(%s, self).__init__()' % (net.motif_name))
        for s in layer_list:
            _str.append('        %s' % (s))
        _str.append('\n    def forward(self, input):')
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.append('        return final')
        _str.append('\n')
        return _str

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

