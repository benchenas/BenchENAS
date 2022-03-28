from __future__ import division
import numpy as np

class PoolUnit():
    
    def __init__(self, _type):
        
        self.type = _type
        self.no = -1
        self.in_size = [0, 0, 0] #width, height
        self.output_size = [0, 0, 0] #width, height
        self.kernel_size = [0, 0]
        self.stride_size = [0, 0]
        self.padding = 0
        
        
    def create(self, no, in_size, kernel_sizde, stride_size, padding):
        self.no = no
        self.in_size = in_size
        self.kernel_size = kernel_sizde
        self.stride_size = stride_size
        self.padding = padding
        self.output_size = self._calculate_output()    
        
        
    def _calculate_output(self):
        # https://blog.csdn.net/zz2230633069/article/details/83214308
        w_in = self.in_size[0]
        h_in = self.in_size[1]
        
        w_out = np.floor((w_in + 2*self.padding - (self.kernel_size[0]-1) - 1)/self.stride_size[0] + 1)
        h_out = np.floor((h_in + 2*self.padding - (self.kernel_size[1]-1) - 1)/self.stride_size[1] + 1)
        
        return [w_out, h_out, self.in_size[2]]
        

        
    def __str__(self): 
        _str = []
        _str.append('%d'%(self.no))
        _str.append('%s'%(self.type))
        _str.append('in:%d-%d-%d'%(self.in_size[0], self.in_size[1], self.in_size[2]))
        _str.append('out:%d-%d-%d'%(self.output_size[0], self.output_size[1], self.output_size[2]))
        _str.append('kernel:%d-%d'%(self.kernel_size[0], self.kernel_size[1]))
        _str.append('stride:%d-%d'%(self.stride_size[0], self.stride_size[1]))
        _str.append('padding:%d'%(self.padding))
        return ','.join(_str)
        
class MaxPool(PoolUnit):
    def __init__(self):
        super().__init__('MAX_POOL')
        
class AvgPool(PoolUnit):
    def __init__(self):
        super().__init__('AVG_POOL')
        