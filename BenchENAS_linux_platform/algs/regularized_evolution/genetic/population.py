import copy
import hashlib
import collections
import numpy as np

Genotype = collections.namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
OPS = ['none',
       'avg_pool_3x3',
       'max_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'sep_conv_7x7',
       'dil_conv_3x3',
       'dil_conv_5x5',
       'conv_7x1_1x7']
NUM_VERTICES = 5


class ArchDarts:
    def __init__(self, arch):
        self.arch = arch

    @classmethod
    def random_arch(cls):
        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(1, len(OPS)), NUM_VERTICES)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
        return (normal, reduction)

    @classmethod
    def transfer_to_genotype(cls, geno_tuple):
        (normal_tup, reduction_tup) = geno_tuple
        normal_concat = list(range(0, 7))
        reduce_concat = list(range(0, 7))
        normal = []
        reduce = []
        for (idx, ops) in normal_tup:
            if idx in normal_concat:
                normal_concat.remove(idx)
            normal.append((OPS[ops], idx))

        for (idx, ops) in reduction_tup:
            if idx in reduce_concat:
                reduce_concat.remove(idx)
            reduce.append((OPS[ops], idx))
        geno = Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)
        return geno


class Individual(object):
    def __init__(self, id, params, tuple=None):
        self.id = id
        self.acc = -1
        self.params = params
        if tuple is None:
            self.tuple = ArchDarts.random_arch()
        else:
            self.tuple = tuple
        self.normal_cell, self.reduction_cell = self.tuple
        self.genotype = ArchDarts.transfer_to_genotype(self.tuple)

    def uuid(self):
        _str1 = 'normal_cell:\n' + str(self.normal_cell)
        _str2 = '\nreduction_cell:\n' + str(self.reduction_cell)
        _str3 = '\ngenotype:\n' + str(self.genotype)
        _str = _str1 + _str2 + _str3
        _final_utf8_str_ = _str.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _str

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        _str.append('normal_cell:' + str(self.normal_cell))
        _str.append('reduction_cell:' + str(self.reduction_cell))
        _str.append('genotype:' + str(self.genotype))
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params):
        self.number_id = 0
        self.pop_size = params['pop_size']
        self.individuals = collections.deque()
        self.params = params

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d'% (self.number_id)
            self.number_id = self.number_id+1
            indi = Individual(indi_no, self.params)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in list(self.individuals):
            _str.append(str(ind))
            _str.append("-"*100)
        return '\n'.join(_str)

    def create_from_offspring(self, offspring):
        self.individuals = offspring

# print(ArchDarts.random_arch())