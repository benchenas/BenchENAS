import numpy as np

class FullConnLayer():
    def __init__(self):
        
        self.type ='FULL'
        self.no = -1
        self.in_size = 0
        self.out_size = 0
        self.dropout = 0
        
        
    def create(self, no, in_size, out_size, dropout):
        self.no = no
        
        self.in_size = in_size if 'int' in type(in_size).__name__ else np.prod(in_size)
        self.out_size = out_size
        self.dropout = dropout
        
    def __str__(self): 
        _str = []
        _str.append('%d'%(self.no))
        _str.append('%s'%(self.type))
        _str.append('in:%d'%(self.in_size))
        _str.append('out:%d'%(self.out_size))
        _str.append('dropout:%.2f'%(self.dropout))
        return ','.join(_str)
    