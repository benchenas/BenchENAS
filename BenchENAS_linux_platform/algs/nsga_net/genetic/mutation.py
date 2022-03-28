import random
import numpy as np
import copy

class Mutation(object):
    def __init__(self, individuals, parent, _log):
        self.individuals = individuals
        self.parent = copy.deepcopy(parent)
        self.log = _log

    def do_mutation_cell(self, cell):
        mut_type = random.random()
        if mut_type < 0.5:
            choose_row = np.random.randint(2, cell.shape[0])
            l = np.nonzero(cell[choose_row])[0]
            if len(l) == 2:
                choosen = l[np.random.randint(0, 2)]
                pre_choosen = cell[choose_row][choosen]
                now_choosen = np.random.randint(1, 7)
                while now_choosen == pre_choosen:
                    now_choosen = np.random.randint(1, 7)
                cell[choose_row][choosen] = now_choosen
            else:
                pre_choosen = cell[choose_row][l[0]]
                op1 = pre_choosen // 10
                op2 = pre_choosen % 10
                if random.random() < 0.5:
                    op_now = np.random.randint(1, 7)
                    while op_now == op1:
                        op_now = np.random.randint(1, 7)
                    op1 = op_now
                else:
                    op_now = np.random.randint(1, 7)
                    while op_now == op2:
                        op_now = np.random.randint(1, 7)
                    op2 = op_now
                if op1 < op2:
                    cell[choose_row][l[0]] = int(str(op1) + str(op2))
                else:
                    cell[choose_row][l[0]] = int(str(op2) + str(op1))
        else:
            choose_row = np.random.randint(2, cell.shape[0])
            l = np.nonzero(cell[choose_row])[0]
            if len(l) == 2:
                c = np.random.randint(0, 2)
                choosen = l[c]
                pre_op = cell[choose_row][choosen]
                cell[choose_row][choosen] = 0
                choose_new = np.random.randint(0, choose_row)
                while choose_new == choosen:
                    choose_new = np.random.randint(0, choose_row)
                if choose_new == l[1 - c]:
                    still_op = cell[choose_row][1 - c]
                    if still_op < pre_op:
                        cell[choose_row][l[0]] = int(str(still_op) + str(pre_op))
                    else:
                        cell[choose_row][l[0]] = int(str(pre_op) + str(still_op))
                else:
                    cell[choose_row][choose_new] = pre_op


    def do_mutation(self):
        after_mut = []
        mut_num = random.random()
        if mut_num < 0.5:
            cell = self.parent.normal_cell
            self.do_mutation_cell(cell)
        else:
            cell = self.parent.reduction_cell
            self.do_mutation_cell(cell)
        self.individuals.append(self.parent)
        return self.individuals

