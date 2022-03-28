import os
import time
import platform
from compute.file import get_algo_local_dir
from algs.cgp_cnn.genetic.population import Population, Individual
from algs.cgp_cnn.genetic.statusupdatetool import StatusUpdateTool
import numpy as np


class Utils(object):
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
    def load_population(cls, prefix, network_info, gen_no):
        file_name = '%s/%s_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), prefix, np.min(gen_no))
        file_name = cls.path_replace(file_name)
        params = StatusUpdateTool.get_init_params()
        pop = Population(gen_no, params, network_info, False)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            gene = np.zeros((network_info.node_num + network_info.out_num, network_info.max_in_num + 1)).astype(int)
            k = 0
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        acc = float(line[4:])
                    elif line.startswith('[['):
                        l = gene.shape[1]
                        sp = line.split('   ', l + 1)
                        sp[0] = sp[0][3:]
                        sp[-1] = sp[-1][0:-1]
                        sp = [int(i) for i in sp]
                        gene[0] = sp
                    elif line.startswith('['):
                        k = k + 1
                        l = gene.shape[1]
                        sp = line.split(' ')
                        while '' in sp:
                            sp.remove('')
                        while '[' in sp:
                            sp.remove('[')
                        if sp[-1][-2] == ']':
                            sp[-1] = sp[-1][0:-2]
                        else:
                            sp[-1] = sp[-1][0:-1]
                        sp = [int(i) for i in sp]
                        gene[k] = sp

            indi = Individual(network_info, False, indi_no)
            indi.eval = acc
            indi.gene = gene
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
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part2.append(line)
            line = f.readline().rstrip()
        return part1, part2

    @classmethod
    def generate_pytorch_file(cls, pop):
        part1, part2 = cls.read_template()
        _in_channel = '        in_channel = %d' % (StatusUpdateTool.get_input_channel())
        _n_class = '        n_class = %d' % (StatusUpdateTool.get_num_class())
        _image_size = '        imgSize = %d' % (StatusUpdateTool.get_imgSize())
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        net = pop.active_net_list()
        _str.append('        cgp = %s' % (str(net)))
        _str.append(_in_channel)
        _str.append(_n_class)
        _str.append(_image_size)
        _str.extend(part2)

        # print('\n'.join(_str))
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), pop.id)
        file_name = cls.path_replace(file_name)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

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
