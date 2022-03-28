from __future__ import division
import numpy as np


class Selection(object):
    def RouletteSelection(self, _a, k):
        a = np.asarray(_a)
        idx = np.argsort(a)
        idx = idx[::-1]
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float)
        selected_index = []
        for i in range(k):
            u = np.random.rand() * sum_a
            sum_ = 0
            for i in range(sort_a.shape[0]):
                sum_ += sort_a[i]
                if sum_ > u:
                    selected_index.append(idx[i])

                    break
        return selected_index



