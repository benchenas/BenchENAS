import copy
import numpy as np
import hashlib
import collections

class Individual(object):
    def __init__(self, id, params, n_var, genome = []):
        self.id = id
        self.acc = -1
        self.flop = -1
        self.params = params
        self.n_var = n_var
        self.rank = np.inf
        self.crowding = -1
        if genome == []:
            self.genome = self.random_generate_genome()
        else:
            self.genome = genome


    def random_generate_genome(self):
        val = np.random.random(self.n_var)
        return (val < 0.5).astype(np.int)

    def uuid(self):
        _str = 'genome:' + str(self.genome)
        _final_utf8_str_ = _str.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _str

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        _str.append('flop:%.5f' % (self.flop))
        _str.append('genome:' + str(self.genome))
        return '\n'.join(_str)

    def reset(self):
        self.acc = -1
        self.flop = -1
        self.rank = np.inf
        self.crowding = -1


class Population(object):
    def __init__(self, gen_no, params):
        self.gen_no = gen_no
        self.number_id = 0
        self.pop_size = params['pop_size']
        self.individuals = []
        self.params = params
        self.n_var = params['n_var']

    def initialize(self):
        for i in range(self.pop_size):
            indi_no = 'indi%05d_%05d'%(self.gen_no, self.number_id)
            self.number_id = self.number_id+1
            indi = Individual(indi_no, self.params, self.n_var)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append("-"*100)
        return '\n'.join(_str)

    def create_from_offspring(self, offspring):
        offs_temp = copy.deepcopy(offspring)
        for indi_ in offs_temp:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id = self.number_id + 1
            self.individuals.append(indi)

