import random
import copy
from algs.regularized_evolution.genetic.population import ArchDarts

class Mutation(object):
    def __init__(self, individual):
        self.individual = individual

    def do_mutation(self):
        indi = self.individual
        mut_type = random.random()
        norm, reduc = indi.tuple
        if mut_type < 0.5: # hidden state mutation
            mut_cell = random.random()
            if mut_cell < 0.5:
                choice = random.choice(list(range(0, len(norm))))
                i = choice // 2 + 2
                idx, ops = norm[choice]
                idx_r = random.choice(list(range(i)))
                while idx_r == idx:
                    idx_r = random.choice(list(range(i)))
                norm[choice] = (idx_r, ops)
            else:
                choice = random.choice(list(range(0, len(reduc))))
                i = choice // 2 + 2
                idx, ops = reduc[choice]
                idx_r = random.choice(list(range(i)))
                while idx_r == idx:
                    idx_r = random.choice(list(range(i)))
                reduc[choice] = (idx_r, ops)
        else:
            mut_cell = random.random()
            if mut_cell < 0.5:
                choice = random.choice(list(range(0, len(norm))))
                idx, ops = norm[choice]
                new_op = random.choice(range(1, 10))
                while new_op == idx:
                    new_op = random.choice(range(1, 10))
                norm[choice] = (idx, new_op)
            else:
                choice = random.choice(list(range(0, len(reduc))))
                idx, ops = reduc[choice]
                new_op = random.choice(range(1, 10))
                while new_op == idx:
                    new_op = random.choice(range(1, 10))
                reduc[choice] = (idx, new_op)
        indi.tuple = (norm, reduc)
        indi.normal_cell, indi.reduction_cell = indi.tuple
        indi.genotype = ArchDarts.transfer_to_genotype(indi.tuple)
        return indi

# from algs.regularized_evolution.genetic.population import Individual
# a = Individual("indi",None)
# print(a.tuple)
# print(Mutation(a).do_mutation().tuple)