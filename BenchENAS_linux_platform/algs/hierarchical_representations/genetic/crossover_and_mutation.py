from deap import tools
import numpy as np
import copy
import random
from algs.hierarchical_representations.genetic.statusupdatetool import StatusUpdateTool

class Mutation(object):
    def __init__(self, individuals,_prob, _log):
        self.individuals = individuals
        self.prob = _prob
        self.log = _log

    def do_mutation(self):
        after_mut = []

        for indi_no in self.individuals:
            indi_tmp = copy.deepcopy(indi_no)
            p_ = random.random()
            if p_ < self.prob:
                self.log.info("mutation hapened!")
                l = np.random.randint(1,StatusUpdateTool.get_L())
                motif_num = len(indi_no.matrixs[l-1])
                motif_choosen_no = np.random.randint(0,motif_num)
                motif_choosen = indi_no.matrixs[l-1][motif_choosen_no]
                num_node = motif_choosen.shape[0]
                i = np.random.randint(0, num_node)
                j = np.random.randint(0, num_node)
                while j==i:
                    j = np.random.randint(0, num_node)
                if i>j:
                    tmp = i
                    i = j
                    j = tmp
                motif_choosen[i][j] = np.random.randint(0, 7)   #改
                indi_tmp.matrixs[l - 1][motif_choosen_no] = motif_choosen
                indi_tmp.acc = -1.0
            else:
                self.log.info("mutation did not hanpen")
            after_mut.append(indi_tmp)
        return after_mut







