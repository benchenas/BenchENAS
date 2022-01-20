from __future__ import division
import numpy as np
class Selection(object):

    def RouletteSelection(self, _a, k):
        '''
        Individuals are selected according to the value of its' acc
        :param _a: list of acc
        :param k: selected_index size
        :return: the selected k index_list
        '''
        a = np.asarray(_a)
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for _ in range(k):
            rand = np.random.random() * sum_a
            sum = 0
            for i in range(len(a)):
                sum += a[i]
                if sum > rand:
                    selected_index.append(i)
                    break

        return selected_index







