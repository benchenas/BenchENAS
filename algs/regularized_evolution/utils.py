import configparser
import os
import platform
import multiprocessing
from compute.file import get_algo_local_dir, get_local_path
import time
import numpy as np
from algs.regularized_evolution.genetic.population import Population, Individual
from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool


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
    def save_population_at_begin(cls, _str, gen_no):
        file_name = '%s/begin_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        # solve the path differences caused by different platforms
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_at_name(cls, _str, gen_no):
        file_name = '%s/%s.txt' % (os.path.join(get_algo_local_dir(), 'populations'), str(gen_no))
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
            matrix_len = params['N'] + 2
            normal_cell = np.zeros((matrix_len, matrix_len)).astype(int)
            reduction_cell = np.zeros((matrix_len, matrix_len)).astype(int)
            k0 = 0
            k1 = 0
            flag = 0
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        acc = float(line[4:])
                    elif line.startswith('normal_cell'):
                        flag = 0
                    elif line.startswith('reduction_cell'):
                        flag = 1
                    elif line.startswith('[['):
                        l = normal_cell.shape[1]
                        sp = line.split('  ', l + 1)
                        sp[0] = sp[0][3:]
                        sp[-1] = sp[-1][0:-1]
                        sp = [int(i) for i in sp]
                        if flag == 0:
                            normal_cell[0] = sp
                        else:
                            reduction_cell[0] = sp
                    elif line.startswith('['):
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
                        if flag == 0:
                            k0 = k0 + 1
                            normal_cell[k0] = sp
                        else:
                            k1 = k1 + 1
                            reduction_cell[k1] = sp

            indi = Individual(indi_no, params, normal_cell_matrix=normal_cell, reduction_cell_matrix=reduction_cell)
            indi.acc = acc
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
        part6 = []
        part7 = []
        f = open(_path)
        f.readline()  # skip this comment

        line = f.readline().rstrip()
        while line.strip() != "#normal_cell_init":
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != "#normal_cell_forward":
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != "#reduction_cell_init":
            part3.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != "#reduction_cell_forward":
            part4.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != "#concat":
            part5.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '#generate_net':
            part6.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '"""':
            part7.append(line)
            line = f.readline().rstrip()
        return part1, part2, part3, part4, part5, part6, part7

    @classmethod
    def add_forward_list(cls, layerlist, forwardlist, t, out_name, in_name, F, kernel_size):
        if t == 1:
            _str = '%s = %s' % (out_name, in_name)
            forwardlist.append(_str)
        elif t == 2:
            conv = 'self.Conv_%s_%s = SeparableConv2d(output_channels, output_channels, %d, padding=%d)' % (
            in_name, out_name, kernel_size, (kernel_size - 1) / 2)
            layerlist.append(conv)
            _str = '%s = self.Conv_%s_%s(%s)' % (out_name, in_name, out_name, in_name)
            forwardlist.append(_str)
        elif t == 3:
            _str = '%s = nn.AvgPool2d(3,1,1)(%s)' % (out_name, in_name)
            forwardlist.append(_str)
        elif t == 4:
            _str = '%s = nn.MaxPool2d(3,1,1)(%s)' % (out_name, in_name)
            forwardlist.append(_str)
        elif t == 5:
            conv = 'self.DilSepConv_%s_%s = SeparableConv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)' % (
            in_name, out_name)
            layerlist.append(conv)
            _str = '%s = self.DilSepConv_%s_%s(%s)' % (out_name, in_name, out_name, in_name)
            forwardlist.append(_str)
        elif t == 6:
            conv = 'self.AsymConv_%s_%s = AsymmetricConv2d(output_channels, output_channels)' % (in_name, out_name)
            layerlist.append(conv)
            _str = '%s = self.AsymConv_%s_%s(%s)' % (out_name, in_name, out_name, in_name)
            forwardlist.append(_str)
        return layerlist, forwardlist

    @classmethod
    def get_layer_list_forward_list(cls, cell_net, F):
        layer_list = []
        forward_list = []
        final_list = []
        for i, node in enumerate(cell_net.without_predecessors):
            _str = '%s = input%d' % (node, i)
            forward_list.append(_str)
        concat_num = 1
        for i, u in enumerate(cell_net.units):
            if u.type < 10:
                layer_list, forward_list = cls.add_forward_list(layer_list, forward_list, u.type, u.out_name, u.in_name,
                                                                F, u.kernel_size)
            else:
                type1 = u.type // 10
                type2 = u.type % 10
                tmp_name1 = u.out_name + "1"
                tmp_name2 = u.out_name + "2"
                layer_list, forward_list = cls.add_forward_list(layer_list, forward_list, type1, tmp_name1, u.in_name,
                                                                F, u.kernel_size)
                layer_list, forward_list = cls.add_forward_list(layer_list, forward_list, type2, tmp_name2, u.in_name,
                                                                F, u.kernel_size)
                _str = "%s = %s+%s" % (u.out_name, tmp_name1, tmp_name2)
                forward_list.append(_str)

            for j, layer in enumerate(cell_net.skipconnections):
                if layer.out_name == u.out_name:
                    layer_list, forward_list = cls.add_forward_list(layer_list, forward_list, layer.type,
                                                                    'residual_' + layer.out_name,
                                                                    layer.in_name, F, layer.kernel_size)
                    _str = '%s = %s + residual_%s' % (layer.out_name, layer.out_name, layer.out_name)
                    forward_list.append(_str)

            for j, node_name in enumerate(cell_net.without_towards):
                if node_name == u.out_name:
                    if len(final_list) == 0:
                        _str = 'final = %s' % (node_name)
                        final_list.append(node_name)
                        forward_list.append(_str)
                    elif len(final_list) > 0:
                        _str = 'final = torch.cat([final,%s],1)' % (node_name)
                        forward_list.append(_str)
                        concat_num = concat_num + 1
                        final_list.append(node_name)
        return layer_list, forward_list, concat_num

    # 1:identity 2:sep conv 3:avg.pool 4.max.pool 5.dil.sep.cpnv 6.1*7then7*1conv
    @classmethod
    def generate_cell_code(cls, cell_net_normal, cell_net_reduction, F, test=False):
        layer_list_normal, forward_list_normal, concat_normal = cls.get_layer_list_forward_list(cell_net_normal, F)
        layer_list_reduction, forward_list_reduction, concat_reduction = cls.get_layer_list_forward_list(
            cell_net_reduction, F)
        layer_list_reduction.append(
            "self.conv_final = SeparableConv2d(output_channels*%d, output_channels*%d, kernel_size=3, stride=2, padding=1, bias=False)" % (
            concat_reduction, concat_reduction))
        forward_concat = []
        forward_concat.append("self.normal_concat = %d" % concat_normal)
        forward_concat.append("self.reduction_concat = %d" % concat_reduction)
        run_net = 'self.net = Net(4, 2, 44, 44, %d)' % (StatusUpdateTool.get_num_class())
        part1, part2, part3, part4, part5, part6, part7 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in layer_list_normal:
            _str.append('        %s' % (s))
        _str.extend(part2)
        for s in forward_list_normal:
            _str.append('        %s' % (s))
        _str.extend(part3)
        for s in layer_list_reduction:
            _str.append('        %s' % (s))
        _str.extend(part4)
        for s in forward_list_reduction:
            _str.append('        %s' % (s))
        _str.extend(part5)
        for s in forward_concat:
            _str.append('        %s' % (s))
        _str.extend(part6)
        _str.append('        %s' % run_net)
        _str.extend(part7)
        if not test:
            file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), cell_net_normal.id)
        else:
            file_name = '%s/regularized_evolution_%s.py' % (os.path.join(get_local_path(), 'examples'), cell_net_normal.id)
        file_name = cls.path_replace(file_name)
        if not os.path.exists(os.path.join(get_algo_local_dir(), 'scripts')):
            os.makedirs(os.path.join(get_algo_local_dir(), 'scripts'))
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def generate_pytorch_file(cls, normal_cell_net, reduction_cell_net, F, test=False):
        cls.generate_cell_code(normal_cell_net, reduction_cell_net, F, test)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()
