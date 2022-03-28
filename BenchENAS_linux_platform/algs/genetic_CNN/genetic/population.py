from scipy import stats
import copy

class Individual(object):
    def __init__(self, id, params, indi=[]):
        self.id = id
        self.acc = -1
        self.cross = -1
        self.l = params['l']
        if len(indi) == 0:
            bernoulliDist = stats.bernoulli(0.5)
            self.indi = list(bernoulliDist.rvs(self.l))
        else:
            self.indi = indi

    def uuid(self):
        _str = ""
        for i in self.indi:
            _str = _str + str(i)
        key = _str
        _str = "["+_str+"]"
        return key, _str

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        s = "["
        for i in self.indi:
            s = s + str(i)
        s += "]"
        _str.append(s)
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
        for indi_ in offspring:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id = self.number_id+1
            self.individuals.append(indi)

    def relabel(self):
        self.number_id = 0
        for i in self.individuals:
            i.id = 'indi%05d_%05d'%(self.gen_no, self.number_id)
            self.number_id += 1

    def recross(self):
        for i in self.individuals:
            i.cross = -1




