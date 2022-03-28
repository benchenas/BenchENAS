from scipy.stats import bernoulli
from algs.hierarchical_representations.genetic.statusupdatetool import StatusUpdateTool
import copy
import numpy as np
import hashlib


class Individual(object):
    def __init__(self, id, params, matrixs=[], only_identity=False):
        self.id = id
        self.acc = -1
        self.level = params['L']
        self.Ms = params['Ms']
        self.Gs = params['Gs']
        if len(matrixs) == 0:
            for i in range(1, self.level):
                matrixs.append([])
                for _ in range(0, self.Ms[i]):
                    if i == 1 and only_identity:
                        matrixs[i - 1].append(self.generate_matrix(self.Gs[i - 1], self.Ms[i - 1], only_identity))
                    else:
                        matrixs[i - 1].append(self.generate_matrix(self.Gs[i - 1], self.Ms[i - 1], False))
            self.matrixs = matrixs
        else:
            self.matrixs = matrixs

    def generate_matrix(self, matrix_len, num_range, only_identity=False):
        m = np.zeros((matrix_len, matrix_len), dtype=np.int)
        if only_identity:
            for i in range(0, matrix_len):
                for j in range(i + 1, matrix_len):
                    m[i][j] = np.random.randint(0, 2, dtype=np.int) * 6
        else:
            for i in range(0, matrix_len):
                for j in range(i + 1, matrix_len):
                    m[i][j] = np.random.randint(0, 2, dtype=np.int) * np.random.randint(0, 7, dtype=np.int)
        return m

    def uuid(self):
        _str = "[" + str(self.matrixs) + "]"
        _final_utf8_str_ = _str.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _str

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        return '\n'.join(_str)


class Population(object):
    def __init__(self, gen_no, params):
        self.gen_no = gen_no
        self.number_id = 0
        self.pop_size = params['pop_size']
        self.individuals = []
        self.params = params

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            self.number_id = self.number_id + 1
            indi = Individual(indi_no, self.params)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append("-" * 100)
        return '\n'.join(_str)

    def create_from_offspring(self, offspring):
        for indi_ in offspring:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id = self.number_id + 1
            self.individuals.append(indi)
