import os
import numpy as np
import time
from algs.evocnn.genetic.population import Population, Individual, ConvUnit, FcUnit, PoolUnit
from compute.file import get_algo_local_dir
from comm.log import Log
import platform
from algs.evocnn.genetic.statusupdatetool import StatusUpdateTool


class Utils(object):
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
                _map[rs_[0]] = float(rs_[1])
            f.close()
        return _map

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc_mean
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
        # solve the path differences caused by different platforms
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
                    if line.startswith('Acc_mean'):
                        indi.acc_mean = float(line[9:])
                    elif line.startswith('Acc_std'):
                        indi.acc_std = float(line[8:])
                    elif line.startswith('Complexity'):
                        indi.complexity = int(line[11:])
                    elif line.startswith('[conv'):
                        data_maps = line[6:-1].split(';')
                        conv_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                conv_params['number'] = int(_value)
                            elif _key == 'filter_size':
                                _value1, _value2 = _value[1:-1].split(',')
                                _value1, _value2 = _value1.strip(), _value2.strip()
                                conv_params['filter_size'] = int(_value1), int(_value2)
                            elif _key == 'in':
                                conv_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                conv_params['out_channel'] = int(_value)
                            elif _key == 'stride_size':
                                _value1, _value2 = _value[1:-1].split(',')
                                _value1, _value2 = _value1.strip(), _value2.strip()
                                conv_params['stride_size'] = int(_value1), int(_value2)
                            elif _key == 'conv_type':
                                if _value.strip() == 'SAME':
                                    conv_params['conv_type'] = 1
                                else:
                                    conv_params['conv_type'] = 0
                            elif _key == 'std':
                                conv_params['std'] = float(_value)
                            elif _key == 'mean':
                                conv_params['mean'] = float(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s' % (_key))
                        conv = ConvUnit(number=conv_params['number'], filter_width=conv_params['filter_size'][0],
                                        filter_height=conv_params['filter_size'][1],
                                        in_channel=conv_params['in_channel'], out_channel=conv_params['out_channel'],
                                        stride_width=conv_params['stride_size'][0],
                                        stride_height=conv_params['stride_size'][1], conv_type=conv_params['conv_type'],
                                        std=conv_params['std'], mean=conv_params['mean'])
                        indi.units.append(conv)
                    elif line.startswith('[fc'):
                        data_maps = line[4:-1].split(';')
                        fc_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                fc_params['number'] = int(_value)
                            elif _key == 'in':
                                fc_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                fc_params['out_channel'] = int(_value)
                            elif _key == 'std':
                                fc_params['std'] = float(_value)
                            elif _key == 'mean':
                                fc_params['mean'] = float(_value)
                            else:
                                raise ValueError('Unknown key for load fc unit, key_name:%s' % (_key))
                        fc = FcUnit(number=fc_params['number'], input_neurons_number=fc_params['in_channel'],
                                    output_neurons_number=fc_params['out_channel'], std=fc_params['std'],
                                    mean=fc_params['mean'])
                        indi.units.append(fc)
                    elif line.startswith('[pool'):
                        pool_params = {}
                        for data_item in line[6:-1].split(';'):
                            _key, _value = data_item.split(':')
                            if _key == 'number':
                                indi.number_id = int(_value)
                                pool_params['number'] = int(_value)
                            elif _key == 'type':
                                pool_params['max_or_avg'] = float(_value)
                            elif _key == 'stride_size':
                                _value1, _value2 = _value[1:-1].split(',')
                                _value1, _value2 = _value1.strip(), _value2.strip()
                                pool_params['stride_size'] = int(_value1), int(_value2)
                            elif _key == 'kernel_size':
                                _value1, _value2 = _value[1:-1].split(',')
                                _value1, _value2 = _value1.strip(), _value2.strip()
                                pool_params['kernel_size'] = int(_value1), int(_value2)
                            else:
                                raise ValueError('Unknown key for load pool unit, key_name:%s' % (_key))
                        pool = PoolUnit(number=pool_params['number'], max_or_avg=pool_params['max_or_avg'],
                                        kernel_width=pool_params['kernel_size'][0],
                                        kernel_height=pool_params['kernel_size'][1],
                                        stride_width=pool_params['stride_size'][0],
                                        stride_height=pool_params['stride_size'][1])
                        indi.units.append(pool)
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

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '# generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '# generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi):
        # query resnet and densenet unit
        unit_list = []
        for index, u in enumerate(indi.units):
            if u.type == 1:
                pad = 'self.pad%d = SamePad2d(kernel_size=(%d, %d), stride=(%d, %d))' % (
                    index, u.filter_size[0], u.filter_size[1], u.stride_size[0], u.stride_size[1])
                unit_list.append(pad)
                layer = 'self.op%d = nn.Conv2d(in_channels=%d, out_channels=%d, kernel_size=(%d, %d), stride=(%d, %d), padding=0)' % (
                    index, u.in_channel, u.out_channel, u.filter_size[0], u.filter_size[1], u.stride_size[0],
                    u.stride_size[1])
                unit_list.append(layer)
                init = 'nn.init.normal_(self.op%d.weight, %f, %f)' % (index, u.mean, u.std)
                unit_list.append(init)
            elif u.type == 3:
                layer = 'self.op%d = nn.Linear(in_features=%d, out_features=%d)' % (
                    index, u.input_neurons_number, u.output_neurons_number)
                unit_list.append(layer)
                init = 'nn.init.normal_(self.op%d.weight, %f, %f)' % (index, u.mean, u.std)
                unit_list.append(init)
        # print('\n'.join(unit_list))

        # query fully-connect layer
        out_channel_list = []
        out_channel_list.append(StatusUpdateTool.get_input_channel())
        image_output_size = StatusUpdateTool.get_input_size()
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            elif u.type == 3:
                out_channel_list.append(u.output_neurons_number)
            else:
                out_channel_list.append(out_channel_list[-1])
                image_output_size = int(image_output_size / u.kernel_size[
                    0])  # default kernel_size = stride_size, and kernel_size[0] = kernel_size[1]
        # print(fully_layer_name, out_channel_list, image_output_size)

        # generate the forward part
        forward_list = []
        is_first_fc = True
        for i, u in enumerate(indi.units):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            if u.type == 1:
                _str = 'out_%d = self.pad%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
                last_out_put = 'out_%d' % i
                _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
            elif u.type == 3:
                if is_first_fc:
                    _str = 'out_%d = out_%d.view(out_%d.size(0), -1)' % (i - 1, i - 1, i - 1)
                    forward_list.append(_str)
                    is_first_fc = False
                _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
            else:
                if u.max_or_avg < 0.5:
                    _str = 'out_%d = F.max_pool2d(%s, %d)' % (
                        i, last_out_put, u.kernel_size[0])  # default kernel_size[0]=kernel_size[1]
                else:
                    _str = 'out_%d = F.avg_pool2d(%s, %d)' % (i, last_out_put, u.kernel_size[0])
                forward_list.append(_str)
        forward_list.append('return out_%d' % (len(indi.units) - 1))
        # print('\n'.join(forward_list))

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('        %s' % ('# all unit'))
        for s in unit_list:
            _str.append('        %s' % (s))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
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


if __name__ == '__main__':
    print()
