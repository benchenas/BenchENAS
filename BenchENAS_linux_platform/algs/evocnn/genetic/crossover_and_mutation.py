import random
import numpy as np
import copy
from algs.evocnn.genetic.population import Individual
from algs.evocnn.utils import Utils
from algs.evocnn.genetic.statusupdatetool import StatusUpdateTool


class CrossoverAndMutation(object):
    def __init__(self, prob_crossover, prob_mutation, _log, individuals, gen_no, _params):
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.individuals = individuals
        self.gen_no = gen_no
        self.crossover_eta = _params['crossover_eta']
        self.mutation_eta = _params['mutation_eta']
        self.acc_mean_threshold = _params['acc_mean_threshold']
        self.complexity_threshold = _params['complexity_threshold']
        self.log = _log
        self.offspring = []

    def process(self):
        crossover = Crossover(self.individuals, self.prob_crossover, self.crossover_eta, self.acc_mean_threshold,
                              self.complexity_threshold, self.log)
        offspring = crossover.do_crossover()
        self.offspring = offspring
        Utils.save_population_after_crossover(self.individuals_to_string(), self.gen_no)

        mutation = Mutation(self.offspring, self.prob_mutation, self.mutation_eta, self.log)
        mutation.do_mutation()

        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%05d_%05d' % (self.gen_no, i)
            indi.id = indi_no

        Utils.save_population_after_mutation(self.individuals_to_string(), self.gen_no)
        return self.offspring

    def individuals_to_string(self):
        _str = []
        for indi in self.offspring:
            _str.append(str(indi))
            _str.append('-' * 100)
        return '\n'.join(_str)


class Crossover(object):
    def __init__(self, individuals, prob_, eta, acc_mean_threshold, complexity_threshold, _log):
        self.individuals = individuals
        self.prob = prob_
        self.eta = eta
        self.acc_mean_threshold = acc_mean_threshold
        self.complexity_threshold = complexity_threshold
        self.log = _log

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = np.random.randint(0, count_)
        idx2 = np.random.randint(0, count_)
        ind1 = self.individuals[idx1]
        ind2 = self.individuals[idx2]

        if ind1.acc_mean > ind2.acc_mean:
            if ind1.acc_mean - ind2.acc_mean > self.acc_mean_threshold:
                winner = ind1
            else:
                if ind2.complexity < (ind1.complexity - self.complexity_threshold):
                    winner = ind2
                else:
                    winner = ind1
        else:
            if ind2.acc_mean - ind1.acc_mean > self.acc_mean_threshold:
                winner = ind2
            else:
                if ind1.complexity < (ind2.complexity - self.complexity_threshold):
                    winner = ind1
                else:
                    winner = ind2
        return winner

    """
    binary tournament selection
    """

    def _choose_two_parents(self):
        # this might choose two same parents
        ind1 = self._choose_one_parent()
        ind2 = self._choose_one_parent()
        return ind1, ind2

    def do_crossover(self):
        new_offspring_list = []
        for _ in range(len(self.individuals) // 2):
            ind1, ind2 = self._choose_two_parents()

            self.log.info('Do crossover on indi:%s and indi:%s' % (ind1.id, ind2.id))
            p1, p2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
            # for different unit, we define two list, one to save their index and the other one save unit
            p1_conv_index_list = []
            p1_conv_layer_list = []
            p1_pool_index_list = []
            p1_pool_layer_list = []
            p1_full_index_list = []
            p1_full_layer_list = []

            p2_conv_index_list = []
            p2_conv_layer_list = []
            p2_pool_index_list = []
            p2_pool_layer_list = []
            p2_full_index_list = []
            p2_full_layer_list = []

            for i in range(len(p1.units)):
                unit = p1.units[i]
                if unit.type == 1:
                    p1_conv_index_list.append(i)
                    p1_conv_layer_list.append(unit)
                elif unit.type == 2:
                    p1_pool_index_list.append(i)
                    p1_pool_layer_list.append(unit)
                else:
                    p1_full_index_list.append(i)
                    p1_full_layer_list.append(unit)

            for i in range(len(p2.units)):
                unit = p2.units[i]
                if unit.type == 1:
                    p2_conv_index_list.append(i)
                    p2_conv_layer_list.append(unit)
                elif unit.type == 2:
                    p2_pool_index_list.append(i)
                    p2_pool_layer_list.append(unit)
                else:
                    p2_full_index_list.append(i)
                    p2_full_layer_list.append(unit)

            # begin crossover on conv layer
            l = min(len(p1_conv_layer_list), len(p2_conv_layer_list))
            for i in range(l):
                unit_p1 = p1_conv_layer_list[i]
                unit_p2 = p2_conv_layer_list[i]
                _p = np.random.random()
                if _p < self.prob:
                    # filter size
                    filter_size_range = StatusUpdateTool.get_conv_filter_size_limit()
                    w1 = unit_p1.filter_size[0]
                    w2 = unit_p2.filter_size[0]
                    n_w1, n_w2 = self.sbx(w1, w2, filter_size_range[0], filter_size_range[1], self.eta)
                    unit_p1.filter_size = int(n_w1), int(n_w1)
                    unit_p2.filter_size = int(n_w2), int(n_w2)
                    # out channel size
                    out_channel_size_range = StatusUpdateTool.get_channel_limit()
                    s1 = unit_p1.out_channel
                    s2 = unit_p2.out_channel
                    n_s1, n_s2 = self.sbx(s1, s2, out_channel_size_range[0], out_channel_size_range[1], self.eta)
                    unit_p1.out_channel = int(n_s1)
                    unit_p2.out_channel = int(n_s2)
                    # mean
                    mean_range = StatusUpdateTool.get_mean_limit()
                    m1 = unit_p1.mean
                    m2 = unit_p2.mean
                    n_m1, n_m2 = self.sbx(m1, m2, mean_range[0], mean_range[1], self.eta)
                    unit_p1.mean = n_m1
                    unit_p2.mean = n_m2
                    # std
                    std_range = StatusUpdateTool.get_std_limit()
                    std1 = unit_p1.std
                    std2 = unit_p2.std
                    n_std1, n_std2 = self.sbx(std1, std2, std_range[0], std_range[1], self.eta)
                    unit_p1.std = n_std1
                    unit_p2.std = n_std2

                p1_conv_layer_list[i] = unit_p1
                p2_conv_layer_list[i] = unit_p2

            # begin crossover on pool layer
            l = min(len(p1_pool_layer_list), len(p2_pool_layer_list))
            for i in range(l):
                unit_p1 = p1_pool_layer_list[i]
                unit_p2 = p2_pool_layer_list[i]
                _p = np.random.random()
                if _p < self.prob:
                    # kernel size
                    pool_kernel_size_range = StatusUpdateTool.get_pool_kernel_size_list()
                    k1 = np.log2(unit_p1.kernel_size[0])
                    k2 = np.log2(unit_p2.kernel_size[0])
                    n_k1, n_k2 = self.sbx(k1, k2, pool_kernel_size_range[0], pool_kernel_size_range[-1], self.eta)
                    n_k1 = int(np.power(2, n_k1))
                    n_k2 = int(np.power(2, n_k2))
                    unit_p1.kernel_size = n_k1, n_k1
                    unit_p2.kernel_size = n_k2, n_k2
                    # pool type
                    t1 = unit_p1.max_or_avg
                    t2 = unit_p2.max_or_avg
                    n_t1, n_t2 = self.sbx(t1, t2, 0, 1, self.eta)
                    unit_p1.max_or_avg = n_t1
                    unit_p2.max_or_avg = n_t2

                p1_pool_layer_list[i] = unit_p1
                p2_pool_layer_list[i] = unit_p2

            # begin crossover on fc layer
            l = min(len(p1_full_layer_list), len(p2_full_layer_list))
            for i in range(l - 1):
                unit_p1 = p1_full_layer_list[i]
                unit_p2 = p2_full_layer_list[i]
                _p = np.random.random()
                if _p < self.prob:
                    # output hidden neurons number
                    hidden_neurons_range = StatusUpdateTool.get_hidden_neurons_limit()
                    n1 = unit_p1.output_neurons_number
                    n2 = unit_p2.output_neurons_number
                    n_n1, n_n2 = self.sbx(n1, n2, hidden_neurons_range[0], hidden_neurons_range[1], self.eta)
                    unit_p1.output_neurons_number = int(n_n1)
                    unit_p2.output_neurons_number = int(n_n2)
                    # mean
                    mean_range = StatusUpdateTool.get_mean_limit()
                    m1 = unit_p1.mean
                    m2 = unit_p2.mean
                    n_m1, n_m2 = self.sbx(m1, m2, mean_range[0], mean_range[1], self.eta)
                    unit_p1.mean = n_m1
                    unit_p2.mean = n_m2
                    # std
                    std_range = StatusUpdateTool.get_std_limit()
                    std1 = unit_p1.std
                    std2 = unit_p2.std
                    n_std1, n_std2 = self.sbx(std1, std2, std_range[0], std_range[1], self.eta)
                    unit_p1.std = n_std1
                    unit_p2.std = n_std2

                p1_full_layer_list[i] = unit_p1
                p2_full_layer_list[i] = unit_p2

            # for the last full layer, only mean and std
            unit_p1 = p1_full_layer_list[-1]
            unit_p2 = p2_full_layer_list[-1]
            _p = np.random.random()
            if _p < self.prob:
                # mean
                mean_range = StatusUpdateTool.get_mean_limit()
                m1 = unit_p1.mean
                m2 = unit_p2.mean
                n_m1, n_m2 = self.sbx(m1, m2, mean_range[0], mean_range[1], self.eta)
                unit_p1.mean = n_m1
                unit_p2.mean = n_m2
                # std
                std_range = StatusUpdateTool.get_std_limit()
                std1 = unit_p1.std
                std2 = unit_p2.std
                n_std1, n_std2 = self.sbx(std1, std2, std_range[0], std_range[-1], self.eta)
                unit_p1.std = n_std1
                unit_p2.std = n_std2
            p1_full_layer_list[-1] = unit_p1
            p2_full_layer_list[-1] = unit_p2

            # assign these crossovered values to the unit_list1 and unit_list2
            unit_list1 = p1.units
            for i in range(len(p1_conv_index_list)):
                unit_list1[p1_conv_index_list[i]] = p1_conv_layer_list[i]
            for i in range(len(p1_pool_index_list)):
                unit_list1[p1_pool_index_list[i]] = p1_pool_layer_list[i]
            for i in range(len(p1_full_index_list)):
                unit_list1[p1_full_index_list[i]] = p1_full_layer_list[i]

            unit_list2 = p2.units
            for i in range(len(p2_conv_index_list)):
                unit_list2[p2_conv_index_list[i]] = p2_conv_layer_list[i]
            for i in range(len(p2_pool_index_list)):
                unit_list2[p2_pool_index_list[i]] = p2_pool_layer_list[i]
            for i in range(len(p2_full_index_list)):
                unit_list2[p2_full_index_list[i]] = p2_full_layer_list[i]

            # re-adjust the in_channel of the above two list
            unit_list1 = Individual.update_all_channels(unit_list1, 0, self.log)
            unit_list2 = Individual.update_all_channels(unit_list2, 0, self.log)

            p1.units = unit_list1
            p2.units = unit_list2
            offspring1, offspring2 = p1, p2
            offspring1.reset_acc()
            offspring2.reset_acc()
            offspring1.complexity = Individual.calculate_complexity(unit_list1)
            offspring2.complexity = Individual.calculate_complexity(unit_list2)
            new_offspring_list.append(offspring1)
            new_offspring_list.append(offspring2)

        self.log.info('CROSSOVER-%d offspring are generated.' % (len(new_offspring_list)))
        return new_offspring_list

    def sbx(self, p1, p2, xl, xu, eta):
        '''
        :param p1: parent1
        :param p2: parent2
        :param xl: minimal
        :param xu: maximal
        :param eta: the parameter of sbx
        :return: two offsprings after crossover
        '''
        # par1 is the greater parent
        if p1 > p2:
            par1 = p1
            par2 = p2
        else:
            par1 = p2
            par2 = p1
        yl = xl
        yu = xu
        rand = np.random.random()
        if rand <= 0.5:
            betaq = (2 * rand) ** (1 / (eta + 1))
        else:
            betaq = (1 / (2 - 2 * rand)) ** (1 / (eta + 1))
        child1 = 0.5 * ((par1 + par2) - betaq * (par1 - par2))
        child2 = 0.5 * ((par1 + par2) + betaq * (par1 - par2))
        if child1 < yl:
            child1 = yl
        if child1 > yu:
            child1 = yu
        if child2 < yl:
            child2 = yl
        if child2 > yu:
            child2 = yu
        return child1, child2


class Mutation(object):
    def __init__(self, individuals, prob_, eta, _log):
        self.individuals = individuals
        self.prob = prob_
        self.eta = eta
        self.log = _log

    def do_mutation(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0, 'ADD': 0, 'REMOVE': 0, 'ALTER': 0}

        mutation_list = StatusUpdateTool.get_mutation_probs_for_each()
        for indi in self.individuals:
            p_ = random.random()
            if p_ < self.prob:
                units_list = []
                is_new = False
                for i in range(len(indi.units) - 1):
                    cur_unit = indi.units[i]
                    p_ = np.random.random()
                    if p_ < 0.5:
                        is_new = True
                        max_length = 6
                        mutation_type = self.select_mutation_type(mutation_list)
                        if mutation_type == 0:
                            current_conv_and_pool_length = indi.get_conv_number() + indi.get_pool_number()
                            if current_conv_and_pool_length < max_length:
                                _stat_param['ADD'] += 1
                                units_list.append(self.generate_a_new_layer(indi, cur_unit.type, len(indi.units)))
                                units_list.append(cur_unit)
                            else:
                                _stat_param['ALTER'] += 1
                                updated_unit = self.alter_a_unit(indi, cur_unit, self.eta)
                                units_list.append(updated_unit)
                        elif mutation_type == 1:
                            _stat_param['ALTER'] += 1
                            updated_unit = self.alter_a_unit(indi, cur_unit, self.eta)
                            units_list.append(updated_unit)
                        elif mutation_type == 2:
                            _stat_param['REMOVE'] += 1
                            # do nothing with units_list
                        else:
                            raise TypeError('Error mutation type :%d, validate range:0-2' % (mutation_type))
                    else:
                        units_list.append(cur_unit)

                # avoid all units have been removed, add a full layer
                if len(units_list) == 0:
                    units_list.append(Individual.init_a_conv(indi))
                    units_list.append(Individual.init_a_pool(indi))
                units_list.append(indi.units[-1])
                # judge the first unit and the second unit
                if units_list[0].type != 1:
                    units_list.insert(0, Individual.init_a_conv(indi))

                if is_new:
                    _stat_param['offspring_new'] += 1
                    units_list = Individual.update_all_channels(units_list, 1, self.log)
                    indi.units = units_list
                    indi.complexity = Individual.calculate_complexity(units_list)
                else:
                    _stat_param['offspring_from_parent'] += 1
            else:
                _stat_param['offspring_from_parent'] += 1
        self.log.info('MUTATION-mutated individuals:%d[ADD:%d,REMOVE:%d,ALTER:%d, no_changes:%d]' % (
            _stat_param['offspring_new'], _stat_param['ADD'], _stat_param['REMOVE'], _stat_param['ALTER'],
            _stat_param['offspring_from_parent']))

    def generate_a_new_layer(self, indi, current_unit_type, unit_length):
        if current_unit_type == 3:
            # judge if current length = 1, add conv or pool
            if unit_length == 1:
                if random.random() < 0.5:
                    return Individual.init_a_conv(indi)
                else:
                    return Individual.init_a_pool(indi)
            else:
                return Individual.init_a_fc(indi)
        else:
            r = random.random()
            if r < 0.5:
                return Individual.init_a_conv(indi)
            else:
                return Individual.init_a_pool(indi)

    def alter_a_unit(self, indi, unit, eta):
        if unit.type == 1:
            # mutate a conv layer
            return self.alter_conv_unit(indi, unit, eta)
        elif unit.type == 2:
            # mutate a pool layer
            return self.alter_pool_unit(indi, unit, eta)
        else:
            # mutate a full layer
            return self.alter_full_layer(indi, unit, eta)

    def alter_conv_unit(self, indi, unit, eta):
        # feature map size, feature map number, mean std
        fms = unit.filter_size[0]
        fmn = unit.out_channel
        mean = unit.mean
        std = unit.std

        new_fms = int(self.pm(indi.min_conv_filter_size, indi.max_conv_filter_size, fms, eta))
        new_fmn = int(self.pm(indi.min_channel, indi.max_channel, fmn, eta))
        new_mean = self.pm(indi.min_mean, indi.max_mean, mean, eta)
        new_std = self.pm(indi.min_std, indi.max_std, std, eta)
        conv_unit = Individual.init_a_conv(indi, _filter_height=new_fms, _filter_width=new_fms, _out_channel=new_fmn,
                                           _mean=new_mean, _std=new_std)
        return conv_unit666

    def alter_pool_unit(self, indi, unit, eta):
        # kernel size, pool_type
        ksize = np.log2(unit.kernel_size[0])
        pool_type = unit.max_or_avg

        new_ksize = self.pm(indi.pool_kernel_size_list[0], indi.pool_kernel_size_list[-1], ksize, eta)
        new_ksize = int(np.power(2, new_ksize))
        new_pool_type = self.pm(0, 1, pool_type, eta)
        pool_unit = Individual.init_a_pool(indi, _kernel_width=new_ksize, _kernel_height=new_ksize,
                                           _max_or_avg=new_pool_type)
        return pool_unit

    def alter_full_layer(self, indi, unit, eta):
        # num of hidden neurons, mean ,std
        n_hidden = unit.output_neurons_number
        mean = unit.mean
        std = unit.std

        new_n_hidden = int(self.pm(indi.min_hidden_neurons, indi.max_hidden_neurons, n_hidden, eta))
        new_mean = self.pm(indi.min_mean, indi.max_mean, mean, eta)
        new_std = self.pm(indi.min_std, indi.max_std, std, eta)
        fc_unit = Individual.init_a_fc(indi, _output_neurons_number=new_n_hidden, _mean=new_mean, _std=new_std)
        return fc_unit

    def select_mutation_type(self, _a):
        a = np.asarray(_a)
        sum_a = np.sum(a).astype(np.float)
        rand = np.random.random() * sum_a
        sum = 0
        mutation_type = -1
        for i in range(len(a)):
            sum += a[i]
            if sum > rand:
                mutation_type = i
                break
        assert mutation_type != -1
        return mutation_type

    def pm(self, xl, xu, x, eta):
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        rand = np.random.random()
        mut_pow = 1.0 / (eta + 1.)
        if rand < 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
            delta_q = 1.0 - val ** mut_pow
        x = x + delta_q * (xu - xl)
        x = min(max(x, xl), xu)
        return x


if __name__ == '__main__':
    cm = CrossoverAndMutation(StatusUpdateTool.get_genetic_probability()[0],
                              StatusUpdateTool.get_genetic_probability()[1], )
