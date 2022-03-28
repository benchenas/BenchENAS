import copy
import numpy as np
import hashlib


class Individual(object):

    def __init__(self, net_info, init, indi_no):
        self.id = indi_no
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.is_pool = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = -1.0
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv()  # In the case of starting only convolution
        else:
            self.init_gene()  # generate initial individual randomly

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.eval))
        _str.append('gene:\n%s' % str(self.gene))
        return '\n'.join(_str)

    def uuid(self):
        # _str = []
        # _str.append('Active_only:%s' % str(self.count_active_node()))
        # _final_str_ = '-'.join(_str)
        _final_str_ = str(self.active_net_list())
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def init_gene_with_conv(self):
        # initial architecture
        arch = ['S_ConvBlock_64_3']

        input_layer_num = int(self.net_info.input_num / self.net_info.rows) + 1
        output_layer_num = int(self.net_info.out_num / self.net_info.rows) + 1
        layer_ids = [((self.net_info.cols - 1 - input_layer_num - output_layer_num) + i) // (len(arch)) for i in
                     range(len(arch))]
        prev_id = 0  # i.e. input layer
        current_layer = input_layer_num
        block_ids = []  # *do not connect with these ids

        # building convolution net
        for i, idx in enumerate(layer_ids):

            current_layer += idx
            n = current_layer * self.net_info.rows + np.random.randint(self.net_info.rows)
            block_ids.append(n)
            self.gene[n][0] = self.net_info.func_type.index(arch[i])
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0

            self.gene[n][1] = prev_id
            for j in range(1, self.net_info.max_in_num):
                self.gene[n][j + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

            prev_id = n + self.net_info.input_num

        # output layer
        n = self.net_info.node_num
        type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
        self.gene[n][0] = np.random.randint(type_num)
        col = np.min((int(n / self.net_info.rows), self.net_info.cols))
        max_connect_id = col * self.net_info.rows + self.net_info.input_num
        min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
            if col - self.net_info.level_back >= 0 else 0

        self.gene[n][1] = prev_id
        for i in range(1, self.net_info.max_in_num):
            self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
        block_ids.append(n)

        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):

            if n in block_ids:
                continue

            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            if n >= self.net_info.node_num:  # output node
                in_num = self.net_info.out_in_num[t]
            else:  # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i + 1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i + 1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def check_pool(self):
        is_pool = True
        pool_num = 0
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if self.is_active[n]:
                if self.gene[n][0] > 19:
                    is_pool = False
                    pool_num += 1
        return is_pool, pool_num

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.01):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        return active_check

    def neutral_mutation(self, mutation_rate=0.01):
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval

    def active_net_list(self):
        self.check_active()
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:  # intermediate node
                    type_str = self.net_info.func_type[t]
                else:  # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i + 1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


class Population(object):
    def __init__(self, gen_no, params, network_info, init = False):
        self.gen_no = gen_no
        self.number_id = 0
        self.pop_size = params['pop_size']
        self.individuals = []
        self.params = params
        self.network_info = network_info
        self.init = init

    def initialize(self):
        indi_no = 'indi%05d_parent' % (self.gen_no)
        indi = Individual(self.network_info, self.init, indi_no)
        self.individuals.append(indi)
        for _ in range(self.params['lam']):
            indi_no = 'indi%05d_%05d'%(self.gen_no, self.number_id)
            self.number_id = self.number_id+1
            indi_tmp = copy.deepcopy(indi)
            indi_tmp.id = indi_no
            self.individuals.append(indi_tmp)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append("-"*100)
        return '\n'.join(_str)

    def create_from_offspring(self, offspring):
        for i, indi_ in enumerate(offspring):
            if i == 0:
                indi = copy.deepcopy(indi_)
                indi_no = 'indi%05d_parent' % (self.gen_no)
                indi.id = indi_no
                self.individuals.append(indi)
            else:
                indi = copy.deepcopy(indi_)
                indi_no = 'indi%05d_%05d'%(self.gen_no, self.number_id)
                indi.id = indi_no
                self.number_id = self.number_id+1
                self.individuals.append(indi)


