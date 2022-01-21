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


if __name__ == '__main__':
    s = Selection()
    a = [1, 2, 9, 8]
    selected_index = s.RouletteSelection(a, k=2000)

    new_a =[a[i] for i in selected_index]
    type1,type2,type8,type9 = 0,0,0,0
    for i in range(2000):
        if new_a[i] == 1:
            type1+=1
        elif new_a[i] == 2:
            type2+=1
        elif new_a[i] == 8:
            type8 += 1
        else:
            type9 += 1

    print(type1,type2,type8,type9)
