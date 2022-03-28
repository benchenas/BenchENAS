import os
import sys


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])


from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
from compute import Config_ini
from comm.log import Log
from comm.utils import GPUFitness
from compute.file import get_algo_local_dir
from algs.regularized_evolution.genetic.population import Population
from algs.regularized_evolution.genetic.evaluate import FitnessEvaluate
from algs.regularized_evolution.genetic.mutation import Mutation
from algs.regularized_evolution.utils import Utils
import collections
import random
import copy
import os


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.history = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(0, self.params)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, self.params, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.acc == -1:
                indi.acc = fitness_map[indi.id]

    def mutation(self, parent):
        cm = Mutation(self.pops.individuals, parent, Log)
        offspring = cm.do_mutation()
        self.history.append(offspring[-1])
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)
        Utils.save_population_after_mutation(str(self.pops), self.pops.gen_no)

    def environment_selection(self):
        v_list = []
        indi_list = []
        _str = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)
            _t_str = 'Indi-%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc)
            _t_str = 'Pare-%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)

        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        self.pops.individuals.popleft()
        offspring = copy.deepcopy(self.pops.individuals)
        next_gen_pops = Population(self.pops.gen_no + 1, self.pops.params)
        next_gen_pops.create_from_offspring(offspring)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%s' % (indi.id, indi.acc, indi.uuid()[0])
            _str.append(_t_str)
        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no - 1)
        Utils.write_to_file('\n'.join(_str), _file)
        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def create_necessary_folders(self):
        sub_folders = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)

    def do_work(self):
        Log.info('*' * 25)
        self.create_necessary_folders()
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % (gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))
        self.history = self.pops.individuals
        while len(self.history) < self.params['cycles']:
            self.params['gen_no'] = gen_no
            self.pops.gen_no = gen_no
            sample = []
            while len(sample) < self.params['sample_size']:
                candidate = random.choice(list(self.pops.individuals))
                sample.append(candidate)
            parent = max(sample, key=lambda i: i.acc)
            # step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (self.pops.gen_no))
            self.mutation(parent)
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (self.pops.gen_no))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (self.pops.gen_no))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (self.pops.gen_no))
            self.history.append(self.pops.individuals[-1])
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (
                    self.pops.gen_no - 1))  # in environment_selection, gen_no increase 1
            gen_no = gen_no + 1
        pop_history = Population(self.pops.gen_no + 1, self.pops.params)
        pop_history.create_from_offspring(self.history)
        Utils.save_population_at_name(str(pop_history), 'history')
        StatusUpdateTool.end_evolution()


class Run(object):
    def __init__(self, alg_list, train_list, gpu_info_list):
        Config_ini.amend(alg_list, train_list, gpu_info_list)
        from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
        StatusUpdateTool.change_cycles(alg_list['max_gen'])

    def do(self):

        from algs.regularized_evolution.utils import Utils
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work()