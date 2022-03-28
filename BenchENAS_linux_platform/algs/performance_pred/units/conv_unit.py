from __future__ import division
import numpy as np


class ConvUnit():
    
    def __init__(self):
        
        self.type ='CONV'
        self.no = -1
        self.in_size = [0, 0, 0] #width, height, channel_size
        self.output_size = [0, 0, 0] #width, height, channel_size
        self.kernel_size = [0, 0]
        self.stride_size = [0, 0]
        self.groups = 0
        self.padding = 0
        self.add_to = []
        self.concatenate_to = [] 
        
    #out_size is the channel number of the output, the size of each output should be calculated
    def create(self, no, in_size, out_size, kernel_size, stride_size, groups, padding, add_to, concatenate_to):
        self.no = no
        self.in_size = in_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.groups = groups
        self.padding = padding
        self.add_to = add_to
        self.concatenate_to = concatenate_to
        self.output_size = self._calculate_output(out_size)
        
    def _calculate_output(self, out_size):
        # https://blog.csdn.net/zz2230633069/article/details/83214308
        w_in = self.in_size[0]
        h_in = self.in_size[1]
        
        w_out = np.floor((w_in + 2*self.padding - (self.kernel_size[0]-1) - 1)/self.stride_size[0] + 1)
        h_out = np.floor((h_in + 2*self.padding - (self.kernel_size[1]-1) - 1)/self.stride_size[1] + 1)
        
        return [w_out, h_out, out_size]
    
    
    def update_output(self, output_size):
        self.output_size = output_size
        
        
    def __str__(self): 
        _str = []
        _str.append('%d'%(self.no))
        _str.append('%s'%(self.type))
        _str.append('in:%d-%d-%d'%(self.in_size[0], self.in_size[1], self.in_size[2]))
        _str.append('out:%d-%d-%d'%(self.output_size[0], self.output_size[1], self.output_size[2]))
        _str.append('kernel:%d-%d'%(self.kernel_size[0], self.kernel_size[1]))
        _str.append('stride:%d-%d'%(self.stride_size[0], self.stride_size[1]))
        _str.append('groups:%d'%(self.groups))
        _str.append('padding:%d'%(self.padding))
        if self.add_to is None or len(self.add_to) == 0:
            _str.append('add_to:%s'%('None'))
        else:
            _str.append('add_to:%s'%('-'.join(list(map(str, self.add_to)))))
        if self.concatenate_to is None or len(self.concatenate_to) == 0:
            _str.append('cat_to:%s'%('None'))
        else:
            _str.append('cat_to:%s'%('-'.join(list(map(str, self.concatenate_to)))))
            
        return ','.join(_str)