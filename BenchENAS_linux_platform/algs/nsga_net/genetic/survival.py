from __future__ import division
import numpy as np

def randomized_argsort(A, method="numpy", order='ascending'):
    if method == "numpy":
        P = np.random.permutation(len(A))
        I = np.argsort(A[P], kind='quicksort')
        I = P[I]

    elif method == "quicksort":
        I = quicksort(A)

    else:
        raise Exception("Randomized sort method not known.")

    if order == 'ascending':
        return I
    elif order == 'descending':
        return np.flip(I, axis=0)
    else:
        raise Exception("Unknown sorting order: ascending or descending.")

def quicksort(A):
    I = np.arange(len(A))
    quicksort(A, I, 0, len(A) - 1)
    return I

def swap(M, a, b):
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


def quicksort(A, I, left, right):
    if left < right:

        index = np.random.randint(left, right + 1)
        swap(I, right, index)

        pivot = A[I[right]]

        i = left - 1

        for j in range(left, right):

            if A[I[j]] <= pivot:
                i += 1
                swap(I, i, j)

        index = i + 1
        swap(I, right, index)

        quicksort(A, I, left, index - 1)
        quicksort(A, I, index + 1, right)


class NonDominatedSorting(object):
    def __init__(self, individuals):
        self.individuals = individuals

    def calc_domination_matrix(self, F, _F=None, epsilon=0.0):

        """
        if G is None or len(G) == 0:
            constr = np.zeros((F.shape[0], F.shape[0]))
        else:
            # consider the constraint violation
            # CV = Problem.calc_constraint_violation(G)
            # constr = (CV < CV) * 1 + (CV > CV) * -1
            CV = Problem.calc_constraint_violation(G)[:, 0]
            constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
        """

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M

    def non_dominated_sort(self, n_stop_if_ranked):
        F = np.full((len(self.individuals), 2), np.nan)
        for i,indi in enumerate(self.individuals):
            F[i][0] = 100 - indi.acc * 100
            F[i][1] = indi.flop
        M = self.calc_domination_matrix(F)

        # calculate the dominance matrix
        n = M.shape[0]

        fronts = []

        if n == 0:
            return fronts

        # final rank that will be returned
        n_ranked = 0
        ranked = np.zeros(n, dtype=np.int)

        # for each individual a list of all individuals that are dominated by this one
        is_dominating = [[] for _ in range(n)]

        # storage for the number of solutions dominated this one
        n_dominated = np.zeros(n)

        current_front = []

        for i in range(n):
            for j in range(i + 1, n):
                rel = M[i, j]
                if rel == 1:
                    is_dominating[i].append(j)
                    n_dominated[j] += 1
                elif rel == -1:
                    is_dominating[j].append(i)
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                current_front.append(i)
                ranked[i] = 1.0
                n_ranked += 1

        # append the first front to the current front
        fronts.append(current_front)

        # while not all solutions are assigned to a pareto front
        while n_ranked < n:
            next_front = []

            # for each individual in the current front
            for i in current_front:

                # all solutions that are dominated by this individuals
                for j in is_dominating[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
                        ranked[j] = 1.0
                        n_ranked += 1
            fronts.append(next_front)
            current_front = next_front
        _fronts = []
        n_ranked = 0
        for front in fronts:

            _fronts.append(np.array(front, dtype=np.int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = _fronts
        return fronts

    def rank_from_fronts(fronts, n):
        # create the rank array and set values
        rank = np.full(n, 1e16, dtype=np.int)
        for i, front in enumerate(fronts):
            rank[front] = i

        return rank


class RankAndCrowdingSurvival(object):

    def __init__(self, indis):
        self.individuals = indis


    def _do(self, n_survive, D=None, **kwargs):

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting(self.individuals).non_dominated_sort(n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            indivis = []
            for i in front:
                indivis.append(self.individuals[i])
            crowding_of_front = calc_crowding_distance(indivis)

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                self.individuals[i].rank = k
                self.individuals[i].crowding = crowding_of_front[j]

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        indivis = []
        for i in survivors:
            indivis.append(self.individuals[i])
        return indivis


def calc_crowding_distance(indis):
    F = np.full((len(indis), 2), np.nan)
    for i, indi in enumerate(indis):
        F[i][0] = 100 - indi.acc
        F[i][1] = indi.flop
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
               - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity

    return crowding


class Survival(object):
    def __init__(self, indis, params):
        self.indivisuals = indis
        self.params = params

    def do(self):
        survival = RankAndCrowdingSurvival(self.indivisuals)._do(self.params['n_survive'])
        return survival

# from algs.nsga_net.genetic.population import Individual
# from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
# indis = []
# params = {}
# for i in range(10):
#     indi1 = Individual("1", params, 5, genome = [])
#     indi1.acc = 0.5
#     indi1.flop = 0.7
#     indis.append(indi1)
#
#     indi2 = Individual("2", params, 5, genome = [])
#     indi2.acc = 0.7
#     indi2.flop = 0.6
#     indis.append(indi2)
#
#     indi3 = Individual("3", params, 5, genome = [])
#     indi3.acc = 0.8
#     indi3.flop = 0.5
#     indis.append(indi3)
#
#
# fronts = RankAndCrowdingSurvival(indis)._do(1000)
# print(fronts)