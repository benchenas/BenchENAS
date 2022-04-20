import copy, random, time, hashlib
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

from algs.large_scale.genetic.statusupdatetool import StatusUpdateTool
from algs.large_scale.utils import Utils
from algs.large_scale.genetic.evaluate import FitnessEvaluate
from compute import Config_ini
from algs.large_scale.genetic.population import Population, ArcText
from algs.large_scale.genetic.mutation import StructMutation
from comm.utils import GPUFitness
from comm.log import Log
from compute.file import get_algo_local_dir


class EvolveCNN(object):
    def __init__(self, params):
        self.pops = None
        self.params = params

    def create_necessary_folders(self):
        sub_folders = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for dna in self.pops.individuals:
            if dna.fitness == -1:
                if dna.individual_id in fitness_map:
                    dna.fitness = fitness_map[dna.individual_id]
                else:
                    dna.fitness = 0.

    def mutate(self):
        mut = StructMutation()
        pop_copy1 = copy.deepcopy(self.pops)
        pop_copy2 = copy.deepcopy(self.pops)
        mutated_pops = pop_copy1.individuals
        for dna in pop_copy2.individuals:
            mutated_dna = dna
            mutated_dna.fitness = -1
            mut.mutate(mutated_dna)
            mutated_pops.append(mutated_dna)
        self.pops.create_from_offspring(mutated_pops)
        Utils.save_population_after_mutation(str(self.pops), self.pops.gen_no)

    def environment_selection(self):
        random.shuffle(self.pops.individuals)
        better_dna = []
        worse_dna = []
        for i, _ in enumerate(self.pops.individuals):
            if i % 2 == 0:
                if self.pops.individuals[i].fitness > self.pops.individuals[i + 1].fitness:
                    better_dna.append(self.pops.individuals[i])
                    worse_dna.append(self.pops.individuals[i + 1])
                else:
                    better_dna.append(self.pops.individuals[i + 1])
                    worse_dna.append(self.pops.individuals[i])
        del worse_dna
        next_gen_pops = Population(self.pops.params, self.pops.gen_no + 1)
        next_gen_pops.create_from_offspring(better_dna)
        self.pops = next_gen_pops
        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def do_work(self, max_gen):
        self.create_necessary_folders()
        self.initialize_population()
        self.fitness_evaluate()
        gen_no = 0
        for i in range(gen_no, max_gen):
            # double individuals thru mutate
            Log.info('EVOLVE[%d-gen]-begin mutate' % self.pops.gen_no)
            self.mutate()
            Log.info('EVOLVE[%d-gen]-end mutate' % self.pops.gen_no)
            # evaluate the fitness
            Log.info('EVOLVE[%d-gen]-begin evaluate' % self.pops.gen_no)
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-end evaluate' % self.pops.gen_no)
            # selection
            Log.info('EVOLVE[%d-gen]-begin selection' % self.pops.gen_no)
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-end selection' % self.pops.gen_no)
            self.pops.gen_no = self.pops.gen_no + 1
        StatusUpdateTool.end_evolution()


class Run(object):
    def __init__(self, alg_list, train_list, gpu_info_list):
        Config_ini.amend(alg_list, train_list, gpu_info_list)

    def do(self):
        from algs.large_scale.genetic.statusupdatetool import StatusUpdateTool
        from algs.large_scale.utils import Utils
        StatusUpdateTool.end_evolution()
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])
