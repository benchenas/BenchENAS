import time
from compute.process import dispatch_to_do
from compute import Config_ini
from compute.gpu import gpus_all_available
from comm.utils import CacheToResultFile
from algs.nsga_net.utils.utils import Utils
from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
from algs.nsga_net.utils.flops_counter import calculate_flop



def decode_generate_file(individual, params, test=False):
    if params['search_space'] == 'micro':
        Utils.generate_micro_pytorch_file(individual, params, test)
    elif params['search_space'] == 'macro':
        channels = [(3, params['init_channels']),
                    (params['init_channels'], 2 * params['init_channels']),
                    (2 * params['init_channels'], 4 * params['init_channels'])]
        Utils.generate_macro_pytorch_file(individual, channels, params, test)


class FitnessEvaluate(object):
    def __init__(self, individuals, params, log):
        self.individuals = individuals
        self.log = log
        self.params = params

    def generate_to_python_file(self, test=False):
        self.log.info("Begin to generate python files")
        for indi in list(self.individuals):
            decode_generate_file(indi, self.params, test)
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
        self.log.info('Total %d cache!' % _count)

        for indi in self.individuals:
            if indi.acc < 0:
                _id = indi.id
                _uuid, _ = indi.uuid()
                dispatch_to_do(_id, _uuid)

        all_have_been_evaluated = False
        while all_have_been_evaluated is not True:
            time.sleep(120)
            all_have_been_evaluated = gpus_all_available()



