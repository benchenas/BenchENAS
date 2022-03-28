from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from algs.hierarchical_representations.genetic.dag import DAG, DAGValidationError
from compute.config import AlgorithmConfig
from algs.hierarchical_representations.utils import Utils
from algs.hierarchical_representations.genetic.statusupdatetool import StatusUpdateTool
from comm.log import Log
from comm.utils import GPUFitness
import time
from compute.file import get_algo_local_dir
from algs.hierarchical_representations.genetic.population import Population
from algs.hierarchical_representations.genetic.evaluate import FitnessEvaluate
from algs.hierarchical_representations.genetic.selection_operator import Selection
from algs.hierarchical_representations.genetic.crossover_and_mutation import Mutation
import numpy as np
import copy
import os


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(0, self.params)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.acc == -1:
                indi.acc = fitness_map[indi.id]

    def mutation(self):
        cm = Mutation(self.pops.individuals, 0.5, Log)
        offspring = cm.do_mutation()
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

        # add log
        # find the largest one's index
        max_index = np.argmax(v_list)
        selection = Selection()
        selected_index_list = selection.RouletteSelection(v_list, k=self.params['pop_size'])
        if max_index not in selected_index_list:
            first_selectd_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selectd_v_list)
            selected_index_list[min_idx] = max_index

        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.gen_no + 1, self.pops.params)
        next_gen_pops.create_from_offspring(next_individuals)

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

    def do_work(self, max_gen):
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
            Log.info('EVOLVE[%d-gen]-Begin to initialize population' % (gen_no))
            self.initialize_population()
            Log.info('EVOLVE[%d-gen]-Finish initialize population' % (gen_no))
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            self.pops.gen_no = curr_gen
            # step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (self.pops.gen_no))
            self.mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (self.pops.gen_no))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (self.pops.gen_no))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (self.pops.gen_no))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (
                    self.pops.gen_no - 1))  # in environment_selection, gen_no increase 1
        StatusUpdateTool.end_evolution()


class Run(object):
    def do(self):
        # StatusUpdateTool.end_evolution()
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])
