from algs.cnn_ga.utils import Utils
import importlib
from multiprocessing import Process
import time, os, sys
from asyncio.tasks import sleep
from comm.utils import CacheToResultFile
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available

class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')

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
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f'%(indi.id, _key, float(_acc), indi.acc))
                CacheToResultFile.do(indi.id, float(_acc))
                indi.acc = float(_acc)

        self.log.info('Total hit %d individuals for fitness'%(_count))

        for indi in self.individuals:
            if indi.acc < 0:
                _id = indi.id
                _uuid, _ = indi.uuid()
                dispatch_to_do(_id, _uuid)

        all_have_been_evaluated = False
        while all_have_been_evaluated is not True:
            # print('All have been evaluated flag ', all_have_been_evaluated)
            time.sleep(60)
            all_have_been_evaluated = gpus_all_available()

        # set the fitness values to each of individual



