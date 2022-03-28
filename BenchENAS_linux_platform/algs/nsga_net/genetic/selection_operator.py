from __future__ import division
import numpy as np
import math


class TournamentSelection(object):
    def __init__(self, indis,pressure=2):
        self.individuals =indis
        self.pressure = pressure

    def do(self, n_select, n_parents=1):
        pop = self.individuals
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = self.random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.binary_tournament(pop, P)

        indivis = []
        choose = np.reshape(S, (n_select, n_parents))
        #print(choose.shape)
        for i in choose:
            tup = []
            for j in i:
                tup.append(pop[j])

            indivis.append(tup)
        return indivis


    def random_permuations(self, n, l):
        perms = []
        for i in range(n):
            perms.append(np.random.permutation(l))
        P = np.concatenate(perms)
        return P

    def compare(self, a, a_val, b, b_val, method, return_random_if_equal=False):
        if method == 'larger_is_better':
            if a_val > b_val:
                return a
            elif a_val < b_val:
                return b
            else:
                if return_random_if_equal:
                    return np.random.choice([a, b])
                else:
                    return None
        elif method == 'smaller_is_better':
            if a_val < b_val:
                return a
            elif a_val > b_val:
                return b
            else:
                if return_random_if_equal:
                    return np.random.choice([a, b])
                else:
                    return None
        else:
            raise Exception("Unknown method.")

    def get_relation(self, a, b, cva=None, cvb=None):

        if cva is not None and cvb is not None:
            if cva < cvb:
                return 1
            elif cvb < cva:
                return -1

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val

    def binary_tournament(self, pop, P):
        if P.shape[1] != 2:
            raise ValueError("Only implemented for binary tournament!")

        tournament_type = 'comp_by_dom_and_crowding'
        S = np.full(P.shape[0], np.nan)

        for i in range(P.shape[0]):

            a, b = P[i, 0], P[i, 1]

            # # if at least one solution is infeasible
            # if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            #     S[i] = self.compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

            # # both solutions are feasible
            # else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = self.get_relation([100-pop[a].acc, pop[a].flop], [100-pop[b].acc, pop[b].flop])
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b
            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = self.compare(a, pop[a].rank, b, pop[b].rank,
                               method='smaller_is_better')
            else:
                raise Exception("Unknown tournament type.")
                # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = self.compare(a, pop[a].crowding, b, pop[b].crowding,
                               method='larger_is_better', return_random_if_equal=True)

        return S[:, None].astype(np.int)

# from algs.nsga_net.genetic.population import Individual
# from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
# indis = []
# params = {}
# for i in range(10):
#     indi1 = Individual("1", params, 5, genome = [])
#     indi1.acc = 0.5
#     indi1.flop = 0.7
#     indi1.rank = 3*i
#     indis.append(indi1)
#
#     indi2 = Individual("2", params, 5, genome = [])
#     indi2.acc = 0.7
#     indi2.flop = 0.6
#     indi2.rank = 3*i+1
#     indis.append(indi2)
#
#     indi3 = Individual("3", params, 5, genome = [])
#     indi3.acc = 0.8
#     indi3.flop = 0.5
#     indi3.rank = 3*i +2
#
#     indis.append(indi3)
# a = (TournamentSelection(indis)._do(10,2))
# print(a)