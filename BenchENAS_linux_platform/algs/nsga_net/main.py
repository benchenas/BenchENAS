from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
from comm.log import Log
from comm.utils import GPUFitness
from compute.file import get_algo_local_dir
from algs.nsga_net.genetic.population import Population
from algs.nsga_net.genetic.evaluate import FitnessEvaluate
from algs.nsga_net.genetic.crossover_and_mutation import CrossoverAndMutation
from algs.nsga_net.utils.utils import Utils
from algs.nsga_net.utils.flops_counter import calculate_flop
from algs.nsga_net.genetic.survival import Survival
from algs.nsga_net.genetic.selection_operator import TournamentSelection
import collections
import random
import numpy as np
import copy
import math
import os


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.parent_pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(0, self.params)
        pops.initialize()
        self.pops = pops
        self.parent_pops = copy.deepcopy(pops)
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, self.params, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            indi.flop = calculate_flop(indi)
            if indi.acc == -1:
                indi.acc = fitness_map[indi.id]
        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def survivial(self):
        self.pops.individuals = Survival(self.pops.individuals, self.params).do()

    def crossover_and_mutation(self, cur):
        cm = CrossoverAndMutation(self.pops.individuals, self.parent_pops, self.params)
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        next_gen_pops = Population(cur, self.pops.params)
        next_gen_pops.create_from_offspring(offspring)
        for indi in next_gen_pops.individuals:
            indi.reset()
        self.pops = next_gen_pops

    def environment_selection(self):
        n_select = math.ceil(self.params['n_offsprings'] / 2)
        self.parent_pops = TournamentSelection(self.pops.individuals).do(n_select, 2)

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
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            self.pops.gen_no = curr_gen
            Log.info('EVOLVE[%d-gen]-Begin to survival' % (curr_gen))
            self.survivial()
            Log.info('EVOLVE[%d-gen]-Finish the survival' % (curr_gen))
            Log.info('EVOLVE[%d-gen]-Begin to selecte' % (curr_gen))
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the selection' % (curr_gen))
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (curr_gen))
            self.crossover_and_mutation(curr_gen)
            Log.info('EVOLVE[%d-gen]-Finish the Crossover and mutation')
            Log.info('EVOLVE[%d-gen]-Begin to evaluate' % (curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])


if __name__ == '__main__':
    r = Run()
    r.do()
