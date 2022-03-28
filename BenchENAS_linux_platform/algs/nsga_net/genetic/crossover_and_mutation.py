import random
import numpy as np
import copy
import math
from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool

def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    for i in range(len(_X)):
        print(X[i][1].genome[M[i]])
        _X[i][0].genome[M[i]] = X[i][1].genome[M[i]]
        _X[i][1].genome[M[i]] = X[i][0].genome[M[i]]

    return _X

class Crossover(object):
    def __init__(self, pop, parent):
        self.pop = pop
        self.parent = parent

    def do(self, prob = 0.5, n_points=2):
        do_crossover = np.random.random(len(self.parent)) < prob

        n_matings = len(self.parent)
        n_var = len(self.parent[0][0].genome)
        # print("n_matings:"+str(n_matings))
        # print("n_var:"+str(n_var))
        # for i in range(n_matings):
        #     print(len(self.parent[i]))

        # start point of crossover
        r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # create for each individual the crossover range
        for i in range(n_matings):

            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, a:b] = True
                j += 2

        _parent = crossover_mask(self.parent, M)
        for i, f in enumerate(do_crossover):
            if f:
                self.parent[i] = _parent[i]

        off = []

        for i in range(0, len(self.parent)):
            off.append(self.parent[i][0])
            off.append(self.parent[i][1])
        return off


class CrossoverAndMutation(object):
    def __init__(self, indis, parents,params):
        self.individuals = indis
        self.params = params
        self.parents = parents

    def process(self):
        pop = self.individuals
        off = Crossover(pop, self.parents).do()
        off = Mutation(off, self.params).do_mutation()
        return off


class Mutation(object):
    def __init__(self, individuals, params, eta=3):
        self.individuals = individuals
        self.eta = float(eta)
        self.params = params


    def do_mutation(self):
        n = len(self.individuals)
        n_var = len(self.individuals[0].genome)
        X = np.zeros((n, n_var))
        for i in range(n):
            X[i] = self.individuals[i].genome

        X = X.astype(np.float)
        Y = np.full((n, n_var), np.inf)

        prob = 1.0 / n_var

        do_mutation = np.random.random((n, n_var)) < prob

        Y[:, :] = X

        xl = np.repeat(self.params['ub'][None, :], n, axis=0)[do_mutation]
        xu = np.repeat(self.params['lb'][None, :], n, axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        indis = copy.deepcopy(self.individuals)
        Y[do_mutation] = _Y
        for i in range(Y.shape[0]):
            indis[i].genome = Y[i].astype(np.int)
        return indis
