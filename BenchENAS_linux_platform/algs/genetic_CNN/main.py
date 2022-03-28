from algs.genetic_CNN.utils import Utils
from algs.genetic_CNN.genetic.statusupdatetool import StatusUpdateTool
from comm.log import Log
from comm.utils import GPUFitness
from compute.file import get_algo_local_dir
from algs.genetic_CNN.genetic.population import Population
from algs.genetic_CNN.genetic.evaluate import FitnessEvaluate
from algs.genetic_CNN.genetic.crossover_and_mutation import Mutation, Crossover
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
                if indi.id in fitness_map:
                    indi.acc = fitness_map[indi.id]
        # for indi in self.pops.individuals:
        #     if indi.acc == -1:
        #         indi.acc = random.random()

    def crossover_and_mutation(self):
        after_cross = Crossover(self.pops.individuals, self.params['crossover_prob'], Log).crossover()
        cm = Mutation(after_cross, self.params['mutation_prob'], Log)
        offspring = cm.do_mutation()
        self.pops.individuals = offspring
        self.pops.relabel()
        self.pops.recross()

    def environment_selection(self, gen):
        individuals = self.pops.individuals
        lowest_acc = 1
        for indi in individuals:
            if indi.acc < lowest_acc:
                lowest_acc = indi.acc
        prob = []
        for indi in individuals:
            prob.append(indi.acc - lowest_acc)
        prob = np.array(prob)
        sum_num = np.sum(prob)
        prob = prob / sum_num
        np.random.seed(0)

        new_individuals = []
        for i in range(0, len(individuals)):
            index = np.random.choice(list(range(0, len(individuals))), p=prob.ravel())
            new_individuals.append(individuals[index])

        pops = Population(gen, self.params)
        pops.individuals = new_individuals
        pops.relabel()
        self.pops = pops
        Utils.save_population_at_begin(str(self.pops), gen)

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
        Utils.save_population_at_evaluation(str(self.pops), gen_no)

        for curr_gen in range(gen_no + 1, max_gen):
            self.params['gen_no'] = curr_gen

            Log.info('EVOLVE[%d-gen]-Begin to do selection' % (curr_gen))
            self.environment_selection(curr_gen)
            Log.info('EVOLVE[%d-gen]-Finish to do selection' % (curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (curr_gen))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))
            Utils.save_population_at_evaluation(str(self.pops), curr_gen)
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        # StatusUpdateTool.end_evolution()
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])


if __name__ == '__main__':
    r = Run()
    r.do()
