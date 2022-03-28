import time
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available
from comm.utils import CacheToResultFile
from algs.nsga_net.utils.utils import Utils
from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
from algs.nsga_net.utils.flops_counter import calculate_flop
import numpy as np


def decode_generate_file(individual, params):
    if params['search_space'] == 'micro':
        Utils.generate_micro_pytorch_file(individual, params)
    elif params['search_space'] == 'macro':
        channels = [(3, params['init_channels']),
                    (params['init_channels'], 2 * params['init_channels']),
                    (2 * params['init_channels'], 4 * params['init_channels'])]
        Utils.generate_macro_pytorch_file(individual, channels, params)


class FitnessEvaluate(object):
    def __init__(self, individuals, params, log):
        self.individuals = individuals
        self.log = log
        self.params = params

    def generate_to_python_file(self):
        self.log.info("Begin to generate python files")
        for indi in list(self.individuals):
            decode_generate_file(indi, self.params)
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

        all_have_been_evaluated = True
        while all_have_been_evaluated is not True:
            time.sleep(120)
            all_have_been_evaluated = gpus_all_available()



# params = {}
# params['N'] = 5
# indi = Individual('id2', params,6, [], [])
#
# # normal_cell_dag, normal_cell_nodes = generate_dag(indi.normal_cell)
# # without_predecessors = normal_cell_dag.ind_nodes()
# # without_successors = normal_cell_dag.all_leaves()
# indi.normal_cell = np.array([[0,0,0,0,0,0,0],
#                     [0,0,0,0,0,0,0],
#                     [12,0,0,0,0,0,0],
#                     [5,0,4,0,0,0,0],
#                     [6,0,2,0,0,0,0],
#                     [0,4,0,0,1,0,0],
#                     [0,0,4,0,0,3,0]])
# print(indi.normal_cell)
# decode_generate_file(indi, 32)
