from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
from comm.log import Log
from comm.utils import GPUFitness
from compute.file import get_algo_local_dir
from algs.regularized_evolution.genetic.population import Population
from algs.regularized_evolution.genetic.evaluate import FitnessEvaluate
from algs.regularized_evolution.genetic.mutation import Mutation
from algs.regularized_evolution.utils import Utils
import collections
import random
import numpy as np
import copy
import os

class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.history = []

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params)
        pops.initialize()
        self.pops = pops

    def fitness_evaluate_child(self,child):
        fitness = FitnessEvaluate([child], self.params, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        if child.acc == -1:
            child.acc = fitness_map[child.id]
        return child

        # for indi in self.pops.individuals:
        #     if indi.acc == -1:
        #         indi.acc = fitness_map[indi.id]

        # for indi in self.pops.individuals:
        #     if indi.acc == -1:
        #         indi.acc = random.random()

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, self.params, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.acc == -1:
                indi.acc = fitness_map[indi.id]

    def mutation(self, parent):
        cm = Mutation(parent).do_mutation()
        cm.acc = -1
        return cm

    def environment_selection(self):
        self.pops.individuals.popleft()

    def create_necessary_folders(self):
        sub_folders  = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)


    def do_work(self):
        Log.info('*'*25)
        self.create_necessary_folders()
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            self.pops = Utils.load_population('population')
            if self.pops is None:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
            else:
                Log.info('Initialize from the existing generation')
            his = Utils.load_population('history')
            if his is None:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
            else:
                Log.info('Initialize from the existing generation')
            self.history = his.individuals
        else:
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE-Begin to evaluate the fitness')
        self.fitness_evaluate()
        Log.info('EVOLVE-Finish the evaluation')
        Utils.save_population_at_begin(str(self.pops))
        Utils.save_file_at_name(str(self.pops), "population")
        self.history = list(copy.deepcopy(self.pops.individuals))
        his_pop = Population(self.params)
        his_pop.create_from_offspring(self.history)
        Utils.save_file_at_name(str(his_pop), "history")


        while len(self.history) < self.params['cycles']:
            sample = []
            while len(sample) < self.params['sample_size']:
                candidate = random.choice(list(self.pops.individuals))
                sample.append(candidate)
            parent = max(sample, key=lambda i: i.acc)
            Log.info('EVOLVE[%d-cycle]-Begin to mutation' % (len(self.history)))
            child = copy.deepcopy(self.mutation(parent))
            Log.info('EVOLVE[%d-cycle]-Finish mutation' % (len(self.history)))

            child.id = 'indi%05d'% (len(self.history))
            Log.info('EVOLVE[%d-cycle]-Begin to evaluate the fitness of child' % len(self.history))
            child = self.fitness_evaluate_child(child)
            Log.info('EVOLVE[%d-cycle]-Finish the evaluation' % (len(self.history)))
            self.pops.individuals.append(child)
            self.history.append(child)
            self.environment_selection()
            Log.info('EVOLVE[%d-cycle]-Finish the selection' % (len(self.history)))
            his_pop = Population(self.params)
            his_pop.create_from_offspring(self.history)
            Utils.save_file_at_name(str(his_pop), "history")
            Utils.save_file_at_name(str(self.pops), "population")
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        # StatusUpdateTool.end_evolution()
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work()


if __name__ == '__main__':
    r = Run()
    r.do()



