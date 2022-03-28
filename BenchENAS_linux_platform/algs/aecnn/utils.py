import os
import numpy as np
import time
from algs.aecnn.genetic.population import Population, Individual, DenseUnit, ResUnit, PoolUnit
from compute.file import get_algo_local_dir
from comm.log import Log
import platform
from algs.aecnn.genetic.statusupdatetool import StatusUpdateTool


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
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('[densenet'):
                        data_maps = line[10:-1].split(',')
                        densenet_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                densenet_params['number'] = int(_value)
                            elif _key == 'amount':
                                densenet_params['amount'] = int(_value)
                            elif _key == 'k':
                                densenet_params['k'] = int(_value)
                            elif _key == 'in':
                                densenet_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                densenet_params['out_channel'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s' % (_key))
                        # get max_input_channel
                        if densenet_params['k'] == 12:
                            rs = StatusUpdateTool.get_densenet_k12()
                            densenet_params['max_input_channel'] = rs[0]
                        elif densenet_params['k'] == 20:
                            rs = StatusUpdateTool.get_densenet_k20()
                            densenet_params['max_input_channel'] = rs[0]
                        elif densenet_params['k'] == 40:
                            rs = StatusUpdateTool.get_densenet_k40()
                            densenet_params['max_input_channel'] = rs[0]
                        densenet = DenseUnit(number=densenet_params['number'], amount=densenet_params['amount'], \
                                             k=densenet_params['k'],
                                             max_input_channel=densenet_params['max_input_channel'], \
                                             in_channel=densenet_params['in_channel'],
                                             out_channel=densenet_params['out_channel'])
                        indi.units.append(densenet)
                    elif line.startswith('[resnet'):
                        data_maps = line[8:-1].split(',')
                        resnet_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                resnet_params['number'] = int(_value)
                            elif _key == 'amount':
                                resnet_params['amount'] = int(_value)
                            elif _key == 'in':
                                resnet_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                resnet_params['out_channel'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s' % (_key))
                        resnet = ResUnit(number=resnet_params['number'], amount=resnet_params['amount'], \
                                         in_channel=resnet_params['in_channel'],
                                         out_channel=resnet_params['out_channel'])
                        indi.units.append(resnet)
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
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi):
        # query resnet and densenet unit
        unit_list = []
        for index, u in enumerate(indi.units):
            if u.type == 1:
                layer = 'self.op%d = ResNetUnit(amount=%d, in_channel=%d, out_channel=%d)' % (
                index, u.amount, u.in_channel, u.out_channel)
                unit_list.append(layer)
            elif u.type == 3:
                layer = 'self.op%d = DenseNetUnit(k=%d, amount=%d, in_channel=%d, out_channel=%d, max_input_channel=%d)' % (
                index, u.k, u.amount, u.in_channel, u.out_channel, u.max_input_channel)
                unit_list.append(layer)
        # print('\n'.join(unit_list))

        # query fully-connect layer
        out_channel_list = []
        image_output_size = StatusUpdateTool.get_input_size()
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            elif u.type == 3:
                out_channel_list.append(u.out_channel)
            else:
                out_channel_list.append(out_channel_list[-1])
                image_output_size = int(image_output_size / 2)
        fully_layer_name = 'self.linear = nn.Linear(%d, %d)' % (
        image_output_size * image_output_size * out_channel_list[-1], StatusUpdateTool.get_num_class())
        # print(fully_layer_name, out_channel_list, image_output_size)

        # generate the forward part
        forward_list = []
        for i, u in enumerate(indi.units):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            if u.type == 1:
                _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
            elif u.type == 3:
                _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)
            else:
                if u.max_or_avg < 0.5:
                    _str = 'out_%d = F.max_pool2d(out_%d, 2)' % (i, i - 1)
                else:
                    _str = 'out_%d = F.avg_pool2d(out_%d, 2)' % (i, i - 1)
                forward_list.append(_str)
        forward_list.append('out = out_%d' % (len(indi.units) - 1))
        # print('\n'.join(forward_list))

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#resnet and densenet unit'))
        for s in unit_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name))

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
    print(StatusUpdateTool.get_init_params())
