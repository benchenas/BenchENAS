import copy
import numpy as np
import hashlib
import collections

class Individual(object):
    def __init__(self, id, params, ops=6, normal_cell_matrix=[], reduction_cell_matrix=[]):
        self.id = id
        self.acc = -1
        self.params = params
        self.N = params['N']
        self.normal_cell = normal_cell_matrix
        self.reduction_cell = reduction_cell_matrix
        self.ops = ops
        if len(self.normal_cell) == 0:
            self.normal_cell = self.generate_cell()
        if len(self.reduction_cell) == 0:
            self.reduction_cell = self.generate_cell()

    def generate_cell(self):
        matrix_len = self.N+2
        matrix = np.zeros((matrix_len, matrix_len), dtype=np.int)
        for i in range(2, matrix_len):
            conn_1 = np.random.randint(0, i)
            conn_2 = np.random.randint(0, i)
            # while conn_2==conn_1:
            #     conn_2 = np.random.randint(0, i)
            if conn_1 == conn_2:
                op1 = np.random.randint(1, self.ops+1)
                op2 = np.random.randint(1, self.ops+1)
                if op1<op2:
                    matrix[i][conn_1] = int(str(op1)+str(op2))
                else:
                    matrix[i][conn_1] = int(str(op2) + str(op1))
            else:
                matrix[i][conn_1] = np.random.randint(1, self.ops+1)
                matrix[i][conn_2] = np.random.randint(1, self.ops+1)
        return matrix

    def uuid(self):
        _str = 'normal_cell:\n' + str(self.normal_cell)
        _str = 'reduction_cell:\n' + str(self.reduction_cell)
        _final_utf8_str_ = _str.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _str

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        _str.append('normal_cell:\n' + str(self.normal_cell))
        _str.append('reduction_cell:\n' + str(self.reduction_cell))
        return '\n'.join(_str)


class Population(object):
    def __init__(self, gen_no, params):
        self.gen_no = gen_no
        self.number_id = 0
        self.pop_size = params['pop_size']
        self.individuals = collections.deque()
        self.params = params

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d_%05d'%(self.gen_no, self.number_id)
            self.number_id = self.number_id+1
            indi = Individual(indi_no, self.params)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append("-"*100)
        return '\n'.join(_str)

    def create_from_offspring(self, offspring):
        offs_temp = copy.deepcopy(offspring)
        while len(offs_temp) != 0:
            indi_ = offs_temp.popleft()
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id = self.number_id + 1
            self.individuals.append(indi)

