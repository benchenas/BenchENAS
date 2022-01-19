import numpy as np
import hashlib
import copy
from algs.aecnn.genetic.statusupdatetool import StatusUpdateTool


class Unit(object):
    def __init__(self, number):
        self.number = number


class ResUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel):  # prob < 0.5
        super().__init__(number)
        self.type = 1
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel


class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg  # max_pool for < 0.5 otherwise avg_pool


class DenseUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel):
        super().__init__(number)
        self.type = 3
        self.amount = amount
        self.k = k
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.max_input_channel = max_input_channel


class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no  # for record the id of current individual
        self.number_id = 0  # for record the latest number of basic unit
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']  # for resnet output

        self.min_resnet = params['min_resnet']  # minimal number of resnet units
        self.max_resnet = params['max_resnet']  # maximal number of resnet units
        self.min_pool = params['min_pool']  # minimal number of pool units
        self.max_pool = params['max_pool']  # maximal number of pool units
        self.min_densenet = params['min_densenet']  # minimal number of densenet units
        self.max_densenet = params['max_densenet']  # manimal number of densenet units

        self.min_resnet_unit = params['min_resnet_unit']
        self.max_resnet_unit = params['max_resnet_unit']

        self.k_list = params['k_list']
        self.min_k12 = params['min_k12']  # minimal number of k_12 for densenet
        self.max_k12 = params['max_k12']
        self.min_k20 = params['min_k20']
        self.max_k20 = params['max_k20']
        self.min_k40 = params['min_k40']
        self.max_k40 = params['max_k40']

        self.max_k12_input_channel = params[
            'max_k12_input_channel']  # if the k is set to 12, its input channel cannot exceed this settings
        self.max_k20_input_channel = params['max_k20_input_channel']
        self.max_k40_input_channel = params['max_k40_input_channel']

        self.units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        # initialize how many resnet unit/pooling layer/densenet unit will be used
        num_resnet = np.random.randint(self.min_resnet, self.max_resnet + 1)
        num_pool = np.random.randint(self.min_pool, self.max_pool + 1)
        num_densenet = np.random.randint(self.min_densenet, self.max_densenet + 1)

        # find the position where the pooling layer can be connected
        total_length = num_resnet + num_pool + num_densenet
        all_positions = np.zeros(total_length, np.int32)
        if num_resnet > 0: all_positions[0:num_resnet] = 1;
        if num_pool > 0: all_positions[num_resnet:num_resnet + num_pool] = 2;
        if num_densenet > 0: all_positions[num_resnet + num_pool:num_resnet + num_pool + num_densenet] = 3;
        for _ in range(10):
            np.random.shuffle(all_positions)
        while all_positions[0] == 2:  # pooling should not be the first unit
            np.random.shuffle(all_positions)

        # initialize the layers based on their positions
        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                resnet = self.init_a_resnet(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None)
                input_channel = resnet.out_channel
                self.units.append(resnet)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)
            elif i == 3:
                densenet = self.init_a_densenet(_number=None, _amount=None, _k=None, _max_input_channel=None,
                                                _in_channel=input_channel)
                input_channel = densenet.out_channel
                self.units.append(densenet)

    """
    Initialize a resnet layer
    """

    def init_a_resnet(self, _number, _amount, _in_channel, _out_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit + 1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
        resnet = ResUnit(number, amount, _in_channel, out_channel)
        return resnet

    def init_a_pool(self, _number, _max_or_avg):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()
        pool = PoolUnit(number, max_or_avg)
        return pool

    def init_a_densenet(self, _number, _amount, _k, _max_input_channel, _in_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d' % (k))
            amount_lower_limit = getattr(self, 'min_k%d' % (k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit + 1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel' % (k))

        true_input = _in_channel
        if true_input > max_input_channel:
            true_input = max_input_channel
        out_channel = true_input + k * amount
        densenet = DenseUnit(number, amount, k, max_input_channel, in_channel=_in_channel, out_channel=out_channel)
        return densenet

    def get_resnet_number(self):
        number = 0
        for unit in self.units:
            if unit.type == 1:
                number += 1
        return number

    def get_pool_number(self):
        number = 0
        for unit in self.units:
            if unit.type == 2:
                number += 1
        return number

    def get_densenet_number(self):
        number = 0
        for unit in self.units:
            if unit.type == 3:
                number += 1
        return number

    @classmethod
    def get_last_output_channel(cls, pos, indi_units):
        '''
        the position varies between [0,len(indi_units)], and 0 denotes the input channel of the individual,
        1 denotes the output channel after one unit
        :param pos: position
        :param indi_units: a units list like self.units
        :return: last output channel before pos
        '''
        last_output_channel = 0
        if pos == 0:
            last_output_channel = StatusUpdateTool.get_input_channel()
        else:
            for i in range(pos - 1, -1, -1):
                if indi_units[i].type == 1 or indi_units[i].type == 3:
                    last_output_channel = indi_units[i].out_channel
                    break
        assert last_output_channel  # return not equal to 0
        return last_output_channel

    @classmethod
    def update_channel_after_pos(cls, pos, indi_units, keep_out_channel, type, log):
        '''
        update the channels after the position
        :param pos: varies between [0,len(indi_units)], and 0 denotes from indi_units[0] to update
        :param indi_units: a units list like self.units
        :param keep_out_channel: last out channel
        :param type: 0 denotes crossover, 1 denotes mutation
        :param log: log in crossover and mutation
        :return: updated indi_units
        '''
        for i in range(pos, len(indi_units)):
            if indi_units[i].type == 1 or indi_units[i].type == 3:
                # generate log
                if type == 0:
                    log.info('Due to the above crossover, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].in_channel, keep_out_channel))
                else:
                    log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].in_channel, keep_out_channel))

                indi_units[i].in_channel = keep_out_channel
                if indi_units[i].type == 1:
                    break
                elif indi_units[i].type == 3:
                    indi_k = indi_units[i].k
                    if indi_k == 12:
                        max_input_channel = StatusUpdateTool.get_densenet_k12()[0]
                    elif indi_k == 20:
                        max_input_channel = StatusUpdateTool.get_densenet_k20()[0]
                    else: # indi_k == 40
                        max_input_channel= StatusUpdateTool.get_densenet_k40()[0]
                    if indi_units[i].in_channel >= max_input_channel:
                        break
                    else:
                        estimated_out_channel = indi_units[i].k * indi_units[i].amount + indi_units[i].in_channel
                        # generate log
                        if type == 0:
                            log.info(
                                'Due to the above crossover, unit at %d changes its output channel from %d to %d' % (
                                    i, indi_units[i].out_channel, estimated_out_channel))
                        else:
                            log.info(
                                'Due to the above mutation, unit at %d changes its output channel from %d to %d' % (
                                    i, indi_units[i].out_channel, estimated_out_channel))

                        indi_units[i].out_channel = estimated_out_channel
                        keep_out_channel = estimated_out_channel
        return indi_units

    def uuid(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
                _pool_type = 0.25 if unit.max_or_avg < 0.5 else 0.75
                _sub_str.append('type:%.2f' % (_pool_type))

            if unit.type == 3:
                _sub_str.append('densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))

            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('type:%.1f' % (unit.max_or_avg))

            if unit.type == 3:
                _sub_str.append('densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))

            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())


def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)


if __name__ == '__main__':
    test_individual()
    # test_population()
