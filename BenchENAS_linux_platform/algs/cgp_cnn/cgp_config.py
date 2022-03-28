# -*- coding: utf-8 -*-
import numpy as np

# network configurations
class CgpInfoConvSet(object):
    def __init__(self, params):
        self.input_num = 1
        # "S_" means that the layer has a convolution layer without downsampling.
        # "D_" means that the layer has a convolution layer with downsampling.
        # "Sum" means that the layer has a skip connection.
        self.func_type = ['S_ConvBlock_32_1',    'S_ConvBlock_32_3',   'S_ConvBlock_32_5',
                          'S_ConvBlock_128_1',    'S_ConvBlock_128_3',   'S_ConvBlock_128_5',
                          'S_ConvBlock_64_1',     'S_ConvBlock_64_3',    'S_ConvBlock_64_5',
                          'S_ResBlock_32_1',     'S_ResBlock_32_3',    'S_ResBlock_32_5',
                          'S_ResBlock_128_1',     'S_ResBlock_128_3',    'S_ResBlock_128_5',
                          'S_ResBlock_64_1',      'S_ResBlock_64_3',     'S_ResBlock_64_5',
                          'Concat', 'Sum',
                          'Max_Pool', 'Avg_Pool']
                          
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            2, 2,
                            1, 1]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = params['rows']
        self.cols = params['cols']
        self.node_num = params['rows'] * params['cols']
        self.level_back = params['level_back']
        self.min_active_num = params['min_active_num']
        self.max_active_num = params['max_active_num']

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
