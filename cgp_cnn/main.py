import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from algs.cgp_cnn.cgp_config import *
from comm.utils import GPUFitness
from comm.log import Log
from compute import Config_ini
from compute.file import get_algo_local_dir

from algs.cgp_cnn.genetic.statusupdatetool import StatusUpdateTool
from algs.cgp_cnn.genetic.population import Population, Individual
from algs.cgp_cnn.genetic.evaluate import FitnessEvaluate
from algs.cgp_cnn.utils import Utils
import numpy as np
import math


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.network_info = CgpInfoConvSet(self.params)
        self.pops = None
        self.init = False
        self.num_eval = 0
        self.max_pool_num = int(math.log2(self.params['imgSize']) - 2)

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(0, self.params, self.network_info, self.init)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.eval == -1 and indi.id in fitness_map.keys():
                indi.eval = fitness_map[indi.id]
        evaluations = np.zeros(len(self.pops.individuals))
        for i in range(len(self.pops.individuals)):
            evaluations[i] = self.pops.individuals[i].eval
        return evaluations

    def mutation(self, gen_no, eval_flag):
        mutation_rate = self.params['mutation_rate']
        for i in range(self.params['lam']):
            eval_flag[i] = False
            self.pops.individuals[i + 1].copy(self.pops.individuals[0])
            active_num = self.pops.individuals[i + 1].count_active_node()
            _, pool_num = self.pops.individuals[i + 1].check_pool()

            while not eval_flag[i] or active_num < self.pops.individuals[
                i + 1].net_info.min_active_num or pool_num > self.max_pool_num:
                self.pops.individuals[i + 1].copy(self.pops.individuals[0])
                eval_flag[i] = self.pops.individuals[i + 1].mutation(mutation_rate)
                active_num = self.pops.individuals[i + 1].count_active_node()
                _, pool_num = self.pops.individuals[i + 1].check_pool()
            self.pops.individuals[i + 1].id = "indi%05d_%05d" % (gen_no, i)
            self.pops.individuals[i + 1].eval = -1
            self.pops.gen_no = gen_no
        self.pops.individuals[0].id = "indi%05d_parent" % (gen_no)
        Utils.save_population_after_mutation(str(self.pops), gen_no)

    def environment_selection(self, evaluations, gen_no):
        mutation_rate = self.params['mutation_rate']
        best_arg = evaluations.argmax()
        if evaluations[best_arg] > self.pops.individuals[0].eval:
            self.pops.individuals[0].copy(self.pops.individuals[best_arg])
        else:
            self.pops.individuals[0].neutral_mutation(mutation_rate)
        self.pops.individuals[0].id = "indi%05d_parent" % (gen_no)
        Utils.save_population_at_begin(str(self.pops), gen_no)

    def create_necessary_folders(self):
        sub_folders = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)

    def do_work(self, max_gen):
        Log.info('*' * 25)
        self.create_necessary_folders()
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % (gen_no))
                print('Initialize from %d-th generation' % (gen_no))
                pops = Utils.load_population('begin', self.network_info, gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()

        eval_flag = np.empty(self.params['lam'])
        active_num = self.pops.individuals[0].count_active_node()
        _, pool_num = self.pops.individuals[0].check_pool()
        if self.init:
            pass
        else:  # in the case of not using an init indiviudal
            while active_num < self.pops.individuals[0].net_info.min_active_num or pool_num > self.max_pool_num:
                self.pops.individuals[0].mutation(1.0)
                active_num = self.pops.individuals[0].count_active_node()
                _, pool_num = self.pops.individuals[0].check_pool()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

        while gen_no < max_gen:
            # reproduction
            Log.info('EVOLVE[%d-gen]-Begin to reproduction/mutation' % (gen_no))
            self.mutation(gen_no, eval_flag)
            Log.info('EVOLVE[%d-gen]-Finish to reproduction/mutation' % (gen_no))
            # evaluation and selection
            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
            evaluations = self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

            self.environment_selection(evaluations, gen_no)
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (gen_no))
            gen_no += 1
        StatusUpdateTool.end_evolution()


class Run(object):
    def __init__(self, alg_list, train_list, gpu_info_list):
        Config_ini.amend(alg_list, train_list, gpu_info_list)

    def do(self):
        from algs.cgp_cnn.genetic.statusupdatetool import StatusUpdateTool
        from algs.cgp_cnn.utils import Utils
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])
