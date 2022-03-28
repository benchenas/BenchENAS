import time
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available
from comm.utils import CacheToResultFile
from algs.regularized_evolution.genetic.dag import DAG, DAGValidationError
from algs.regularized_evolution.utils import Utils
from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
import numpy as np


def generate_dag(matrix):
    # create nodes for the graph
    l = matrix.shape[0]
    nodes = np.empty((0), dtype=np.str)
    for n in range(0, l):
        nodes = np.append(nodes, ''.join(["node", "_", str(n)]))
    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    for i in range(2, l):
        for j in range(0, i):
            if matrix[i][j] != 0:
                dag.add_edge(''.join(["node", "_", str(j)]), ''.join(["node", "_", str(i)]))
                dag.add_edge_type(''.join(["node", "_", str(j)]), ''.join(["node", "_", str(i)]), matrix[i][j])
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
        if type == 2:
            conv_sizes = StatusUpdateTool.get_conv_size()
            conv_size = conv_sizes[np.random.randint(0, len(conv_sizes))]
            self.kernel_size = conv_size
        else:
            self.kernel_size = 3


class Network(object):
    def __init__(self, id):
        self.units = []
        self.skipconnections = []
        self.without_towards = []
        self.without_predecessors = []
        self.id = id

    def add_edge(self, in_name, out_name, type):
        conv = layer(in_name, out_name, type)
        self.units.append(conv)

    def add_skip(self, ind_node_name, dep_node_name, ops_type):
        conv = layer(ind_node_name, dep_node_name, ops_type)
        self.skipconnections.append(conv)

    def add_without_predecessors(self, node_name):
        self.without_predecessors.append(node_name)

    def add_without_towards(self, node_name):
        self.without_towards.append(node_name)


def get_net(cell, id):
    normal_cell_dag, normal_cell_nodes = generate_dag(cell)
    without_predecessors = normal_cell_dag.ind_nodes()
    without_successors = normal_cell_dag.all_leaves()
    net = Network(id)
    for wop in without_predecessors:
        net.add_without_predecessors(wop)
    for n in normal_cell_nodes:
        predecessors = normal_cell_dag.predecessors(n)
        if len(predecessors) == 0:
            continue
        elif len(predecessors) > 1:
            for prd in range(1, len(predecessors)):
                net.add_skip(predecessors[prd], n, normal_cell_dag.type[predecessors[prd] + "_" + n])
            net.add_edge(predecessors[0], n, normal_cell_dag.type[predecessors[0] + "_" + n])
        elif len(predecessors) == 1:
            net.add_edge(predecessors[0], n, normal_cell_dag.type[predecessors[0] + "_" + n])
    if len(without_successors) > 0:
        for suc in range(0, len(without_successors)):
            net.add_without_towards(without_successors[suc])
    return net


def decode_generate_file(individual, F, test=False):
    normal_cell_net = get_net(individual.normal_cell, individual.id)
    reduction_cell_net = get_net(individual.reduction_cell, individual.id)
    Utils.generate_pytorch_file(normal_cell_net, reduction_cell_net, F, test)


class FitnessEvaluate(object):
    def __init__(self, individuals, params, log):
        self.individuals = list(individuals)
        self.params = params
        self.log = log

    def generate_to_python_file(self, test=False):
        self.log.info("Begin to generate python files")
        for indi in list(self.individuals):
            decode_generate_file(indi, self.params['F'], test)
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


