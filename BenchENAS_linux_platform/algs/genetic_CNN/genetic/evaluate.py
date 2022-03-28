import time
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available
from comm.utils import CacheToResultFile
from algs.genetic_CNN.genetic.dag import DAG
from algs.genetic_CNN.utils import Utils
from algs.genetic_CNN.genetic.statusupdatetool import StatusUpdateTool
import numpy as np


def has_same_elements(x):
    return len(set(x)) <= 1


def generate_dag(optimal_indvidual, stage_name, num_nodes):
    # create nodes for the graph
    nodes = np.empty((0), dtype=np.str)
    for n in range(1, (num_nodes + 1)):
        nodes = np.append(nodes, ''.join([stage_name, "_", str(n)]))

    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    # split best indvidual found via GA to identify vertices connections and connect them in DAG
    edges = np.split(optimal_indvidual, np.cumsum(range(num_nodes - 1)))[1:]
    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(''.join([stage_name, "_", str(v1)]), ''.join([stage_name, "_", str(v2)]))
            v1 += 1
        v2 += 1

    # delete nodes not connected to anyother node from DAG
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes


class conv_layer(object):
    def __init__(self, input_channel, output_channel, in_name, out_name, kernel_size=3, stride_size=1, in_nodes=1):
        self.type = 0
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_name = in_name
        self.out_name = out_name
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.in_nodes = in_nodes


class pool_layer(object):
    def __init__(self, out_name="input", kernel_size=2, stride_size=2):
        self.type = 1
        self.out_name = out_name
        self.kernel_size = kernel_size
        self.stride_size = stride_size


class Network(object):
    def __init__(self, id):
        self.id = id
        self.units = []
        self.skipconnections = []
        self.without_towards = []

    def add_node(self, input_channel, output_channel, in_name, out_name, kernel_size=3, stride_size=1, in_nodes=1):
        conv = conv_layer(input_channel, output_channel, in_name, out_name, kernel_size, stride_size, in_nodes)
        self.units.append(conv)

    def add_pool(self, name, kernel_size=2, stride_size=2):
        pool = pool_layer(name, kernel_size, stride_size)
        self.units.append(pool)

    def add_skip(self, ind_node_name, dep_node_name):
        self.skipconnections.append([ind_node_name, dep_node_name])

    def add_without_towards(self, node_name):
        self.without_towards.append(node_name)


def decode_generate_file(individual):
    net = Network(individual.id)
    stages = StatusUpdateTool.get_stages()
    num_nodes = StatusUpdateTool.get_num_nodes()
    L, BITS_INDICES, _ = StatusUpdateTool.get_params()
    ic = StatusUpdateTool.get_input_size()[2]
    oc = 20
    for stage_index, stage_name, num_node, bpi in zip(range(0, len(stages)), stages, num_nodes, BITS_INDICES):
        indv = individual.indi[bpi[0]:bpi[1]]
        if stage_index == 0:
            net.add_node(ic, oc, 'input', ''.join([stage_name, "_input"]))
            ic = 20
            oc = 20
        elif stage_index == 1:
            net.add_node(20, 50, 'input', ''.join([stage_name, "_input"]))
            ic = 50
            oc = 50
        else:
            net.add_node(ic, oc, 'input', ''.join([stage_name, "_input"]))
            ic = 50
            oc = 50

        if not has_same_elements(indv):
            dag, nodes = generate_dag(indv, stage_name, num_node)
            without_predecessors = dag.ind_nodes()
            without_successors = dag.all_leaves()
            for wop in without_predecessors:
                net.add_node(ic, oc, ''.join([stage_name, "_input"]), wop)
            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) > 1:
                    for prd in range(1, len(predecessors)):
                        net.add_skip(predecessors[prd], n)
                    net.add_node(ic, oc, predecessors[0], n, in_nodes=len(predecessors))
                elif len(predecessors) == 1:
                    net.add_node(ic, oc, predecessors[0], n)
            if len(without_successors) > 0:
                for suc in range(0, len(without_successors)):
                    net.add_without_towards(without_successors[suc])
            net.add_node(ic, oc, ''.join(['final_', stage_name]), 'input')
            net.add_pool('input', 2, 2)
        else:
            net.add_pool(''.join([stage_name, "_input"]), 2, 2)
    Utils.generate_pytorch_file(net)


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
            if indi.acc > 0:
                CacheToResultFile.do(indi.id, float(indi.acc))
            else:
                _key, _str = indi.uuid()
                if _key in _map:
                    _count += 1
                    _acc = _map[_key]
                    self.log.info('Hit the cache for %s, key:%s, acc:%.5f' % (indi.id, _key, float(_acc)))
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
