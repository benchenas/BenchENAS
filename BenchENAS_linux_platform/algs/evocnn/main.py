import numpy as np
import copy
import os
from compute.file import get_algo_local_dir
from comm.log import Log
from comm.utils import GPUFitness
from algs.evocnn.utils import Utils
from algs.evocnn.genetic.statusupdatetool import StatusUpdateTool
from algs.evocnn.genetic.population import Population
from algs.evocnn.genetic.evaluate import FitnessEvaluate
from algs.evocnn.genetic.crossover_and_mutation import CrossoverAndMutation
from algs.evocnn.genetic.selection_operator import Selection


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        # for indi in self.pops.individuals:
        #     if indi.acc_mean == -1:
        #         indi.acc_mean = np.random.random()
        fitness.evaluate()
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.acc_mean == -1:
                indi.acc_mean = fitness_map[indi.id]

    def crossover_and_mutation(self):
        params = {}
        params['crossover_eta'] = StatusUpdateTool.get_crossover_eta()
        params['mutation_eta'] = StatusUpdateTool.get_mutation_eta()
        params['acc_mean_threshold'] = StatusUpdateTool.get_acc_mean_threshold()
        params['complexity_threshold'] = StatusUpdateTool.get_complexity_threshold()
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log,
                                  self.pops.individuals, self.pops.gen_no, params)
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        elite_rate = 0.2
        e_count = int(np.floor(len(self.pops.individuals) * elite_rate / 2) * 2)
        v_list = []
        indi_list = []
        _str = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc_mean)
            _t_str = 'Indi-%s-%.5f-%s' % (indi.id, indi.acc_mean, indi.uuid()[0])
            _str.append(_t_str)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.acc_mean)
            _t_str = 'Pare-%s-%.5f-%s' % (indi.id, indi.acc_mean, indi.uuid()[0])
            _str.append(_t_str)

        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        # add log
        # find the elite's index
        elite_index = np.argsort(v_list)
        elite_index = elite_index[::-1]
        selection = Selection()
        selected_index_list = selection.RouletteSelection(v_list, k=self.params['pop_size'])
        first_selectd_v_list = [v_list[i] for i in selected_index_list]
        sort_first_v_list_index = np.argsort(first_selectd_v_list)
        for i in range(e_count):
            index = elite_index[i]
            if index not in selected_index_list:
                selected_index_list[sort_first_v_list_index[i]] = index

        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no + 1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%s' % (indi.id, indi.acc_mean, indi.uuid()[0])
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
        # create the corresponding fold under runtime
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
            # step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (self.pops.gen_no))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (self.pops.gen_no))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (self.pops.gen_no))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (self.pops.gen_no))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (
                    self.pops.gen_no - 1))  # in environment_selection, gen_no increase 1
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveCNN(params)
        evoCNN.do_work(params['max_gen'])


if __name__ == '__main__':
    r = Run()
    r.do()
    # params = StatusUpdateTool.get_init_params()
    # evoCNN = EvolveCNN(params)
    # evoCNN.create_necessary_folders()
    # evoCNN.initialize_population()
    # evoCNN.pops = Utils.load_population('begin', 0)
    # evoCNN.fitness_evaluate()
    # evoCNN.crossover_and_mutation()
