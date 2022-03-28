import time
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available
from comm.utils import CacheToResultFile
from algs.hierarchical_representations.utils import Utils
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from algs.hierarchical_representations.genetic.dag import DAG, DAGValidationError
from algs.hierarchical_representations.utils import Utils
from algs.hierarchical_representations.genetic.statusupdatetool import StatusUpdateTool
from comm.log import Log
from comm.utils import GPUFitness
import numpy as np
import copy
import os


def has_same_elements(x):
    return len(set(x)) <= 1


def generate_dag(matrix, level):
    # create nodes for the graph
    l = matrix.shape[0]
    nodes = np.empty((0), dtype=np.str)
    for n in range(1, (l + 1)):
        nodes = np.append(nodes, ''.join(["level" + str(level), "_", str(n)]))

    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    for i in range(0, l):
        for j in range(1, l):
            if matrix[i][j] != 0:
                dag.add_edge(''.join(["level" + str(level), "_", str(i + 1)]),
                             ''.join(["level" + str(level), "_", str(j + 1)]))
                dag.add_edge_type(''.join(["level" + str(level), "_", str(i + 1)]),
                                  ''.join(["level" + str(level), "_", str(j + 1)]),
                                  "motif_" + str(level) + "_" + str(matrix[i][j]))
    # delete nodes not connected to anyother node from DAG
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes


class layer(object):
    def __init__(self, in_name, out_name, type):
        self.type = type
        self.in_name = in_name
        self.out_name = out_name


class Network(object):
    def __init__(self, id, level):
        self.units = []
        self.skipconnections = []
        self.without_towards = []
        self.without_predecessors = []
        self.id = id
        self.motif_name = level

    def add_node(self, in_name, out_name, type):
        conv = layer(in_name, out_name, type)
        self.units.append(conv)

    def add_skip(self, ind_node_name, dep_node_name):
        self.skipconnections.append([ind_node_name, dep_node_name])

    def add_without_predecessors(self, node_name):
        self.without_predecessors.append(node_name)

    def add_without_towards(self, node_name):
        self.without_towards.append(node_name)


def decode_generate_file(individual):
    motif_str = []
    for i in range(1, individual.level):
        for k, matrix in enumerate(individual.matrixs[i - 1]):
            dag, nodes = generate_dag(matrix, i)
            without_predecessors = dag.ind_nodes()
            without_successors = dag.all_leaves()
            net = Network(individual.id, "motif_" + str(i + 1) + "_" + str(k + 1))
            for wop in without_predecessors:
                net.add_without_predecessors(wop)
            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) > 1:
                    for prd in range(1, len(predecessors)):
                        net.add_skip(predecessors[prd], n)
                    net.add_node(predecessors[0], n, dag.type[predecessors[0] + "_" + n])
                elif len(predecessors) == 1:
                    net.add_node(predecessors[0], n, dag.type[predecessors[0] + "_" + n])
            if len(without_successors) > 0:
                for suc in range(0, len(without_successors)):
                    net.add_without_towards(without_successors[suc])
            motif_str.append(Utils.generate_layer(net))
    Utils.generate_all(individual, motif_str)


class FitnessEvaluate(object):
    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info("Begin to generate python files")
        for indi in self.individuals:
            decode_generate_file(indi)
        self.log.info("Finished the generation of python files")

    def evaluate(self):
        """
        load fitness from cache file
        """
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f' % (_key, _key, float(_acc)))
                CacheToResultFile.do(indi.id, float(_acc))
                indi.acc = float(_acc)

        for indi in self.individuals:
            if indi.acc < 0:
                _id = indi.id
                _uuid, _ = indi.uuid()
                dispatch_to_do(_id, _uuid)

        all_have_been_evaluated = False
        while all_have_been_evaluated is not True:
            time.sleep(120)
            all_have_been_evaluated = gpus_all_available()
