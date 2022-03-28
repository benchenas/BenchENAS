from algs.performance_pred.units.conv_unit import ConvUnit
from algs.performance_pred.units.pool_unit import MaxPool, AvgPool
from algs.performance_pred.units.full_layer import FullConnLayer
import numpy as np
import hashlib

class CNN():
    
    def __init__(self, _id, params):
        
        self.id = 'indi%05d'%(_id)
        self.acc = -1
        #settings for the main data
        self.input_size = params['input_size']
        
        #settings for the whole architecture
        self.max_length = params['max_length']
        self.min_length = params['min_length']
        self.pool_amount =params['pool_amount']
        self.pool_amount_prob = params['pool_amount_prob']
        self.pool_type_prob = params['pool_type_prob'] # first for max, second for avg
        self.full_amount = params['full_amount']
        self.full_amount_prob = params['full_amount_prob']
        
        #settings for the convolution units
        self.conv_output_channel = params['conv_output_channel']
        self.conv_output_channel_prob = params['conv_output_channel_prob']
        
        self.conv_kernel = params['conv_kernel']
        self.conv_kernel_prob = params['conv_kernel_prob'] # 0 means the average probability for each candidate
        
        self.conv_stride = params['conv_stride']
        self.conv_stride_prob = params['conv_stride_prob']
        
        self.conv_groups = params['conv_groups']
        self.conv_groups_prob = params['conv_groups_prob']
        
        self.conv_padding = params['conv_padding']
        self.conv_padding_prob = params['conv_padding_prob']
        
        self.conv_add_to_prob = params['conv_add_to_prob']
        self.conv_concatenate_to_prob = params['conv_concatenate_to_prob']
        
        #settings for the pooling units
        self.pool_kernel = params['pool_kernel']
        self.pool_kernel_prob = params['pool_kernel_prob']
        self.pool_stride = params['pool_stride']
        self.pool_stride_prob = params['pool_stride_prob']
        self.pool_padding = params['pool_padding']
        self.pool_padding_prob = params['pool_padding_prob']
        
        #settings for the full connection units
        self.full_size = params['full_size']
        self.full_size_prob = params['full_size_prob']
        self.full_dropout = params['full_dropout']
        self.full_dropout_prob = params['full_dropout_prob']
        
        self.units = []
        self.pool_unit_size = 0
        self.full_unit_size = 0
        self.conv_unit_size = 0
        self.network_depth = 0
        
        
    def create(self):
        network_depth = np.random.randint(self.min_length, self.max_length+1)
        pool_unit_size = self.sample_result(self.pool_amount_prob, self.pool_amount)
        full_unit_size = self.sample_result(self.full_amount_prob, self.full_amount)
        conv_unit_size = network_depth - pool_unit_size - full_unit_size
        
        self.network_depth = network_depth
        self.conv_unit_size = conv_unit_size
        self.pool_unit_size = pool_unit_size
        self.full_unit_size = full_unit_size
        
        # find the pooling unit positions, the left positions are for convolutional units
        flag_list = [i for i in range(network_depth-full_unit_size)]
        assert pool_unit_size>0, 'The number of pooling units must be greater than zeros'
        pool_unit_positions = np.random.choice(range(1, len(flag_list)), pool_unit_size)
        for i in pool_unit_positions:
            flag_list[i] = -1 #denote the current position holds a pooling unit
        assert flag_list[0] > -1, 'The pooling layer cannot be the first unit of a CNN'
        
        input_size = self.input_size
        for i in range(len(flag_list)):
            if flag_list[i] > -1:
                conv = self.init_conv(i, input_size)
                input_size = conv.output_size
                self.units.append(conv)
            else:
                if input_size[0] <=2: # the input size is not greater than 2, so the pooling should not be performed
                    conv = self.init_conv(i, input_size)
                    input_size = conv.output_size
                    self.units.append(conv)
                    
                    self.conv_unit_size = self.conv_unit_size + 1
                    self.pool_unit_size = self.pool_unit_size - 1
                else:    
                    pool = self.init_pool(i, input_size)
                    input_size = pool.output_size
                    self.units.append(pool)
                    
        #add jump connection between each block separated by pool units        
        blocks = []
        single_block = []
        for i in range(len(flag_list)):
            if flag_list[i] > -1:
                single_block.append(i)
            else:
                blocks.append(single_block)
                single_block = []
        blocks.append(single_block)
        
        
        for each_block in blocks:
            if len(each_block)>2:
                add_to = self.create_skip(each_block)
                concatenate_to = self.create_skip(each_block)
                if add_to is not None:
                    for each_from_id in add_to.keys():
                        self.units[each_from_id].add_to = add_to[each_from_id]
                if concatenate_to is not None:
                    for each_from_id in concatenate_to.keys():
                        self.units[each_from_id].concatenate_to = concatenate_to[each_from_id]
        
        
        # for the fully connection layer
        for j in range(i+1, i+full_unit_size+1):
            full_unit = self.init_full(j, input_size)
            input_size = full_unit.out_size
            self.units.append(full_unit)
                
    def create_skip(self, block_ids):
        #choose how many units where the skips are from
        from_numbers = np.random.choice(len(block_ids)-1)
        if from_numbers > 0:
            copy_block_ids = [i for i in block_ids]
            del copy_block_ids[-1]
            from_ids = np.random.choice(copy_block_ids, from_numbers)
            effective_from_ids = list(set(from_ids))
            end_id = block_ids[-1]
            
            #find the to-units for each from-id
            rs = {}
            for each_from_id in effective_from_ids:
                to_numbers = np.random.choice(end_id-each_from_id)
                if to_numbers > 0:
                    this_to_list = np.random.choice(range(each_from_id+2, end_id+1), to_numbers)
                    this_to_list = list(set(this_to_list))
                    rs[each_from_id] = this_to_list
#             print(rs)
#             print(rs.keys(), len(rs.keys()))
#             for from_id in rs.keys():
#                 print(from_id)
            if len(rs.keys()) == 0:
                return None
            else:
                return rs
        else:
            return None
        
        
        
        
    
    def init_conv(self, _no, _in_size):
        out_size = self.sample_result(self.conv_output_channel_prob, self.conv_output_channel)
        if _in_size[0] <=2:
            kernel_size = 1
            stride_size = 1
        else:
            kernel_size = self.sample_result(self.conv_kernel_prob, self.conv_kernel)
            stride_size = self.sample_result(self.conv_stride_prob, self.conv_stride)

        groups = self.sample_result(self.conv_groups_prob, self.conv_groups)
        '''
        Based on the design, the kernel can be chosen from {1,3,5,7} while the stride is chosen from {1}
        Therefore, the padding can be calculated as (kernel-stride)/2
        '''
        padding = np.int((kernel_size-stride_size)/2)
        
        conv = ConvUnit()
        conv.create(no=_no, in_size=_in_size, out_size=out_size, kernel_size=[kernel_size,kernel_size], stride_size=[stride_size,stride_size], groups=groups, padding=padding, add_to=None, concatenate_to=None)
        return conv
    
    def init_pool(self, _no, _in_size):
        kernel_sizde = self.sample_result(self.pool_kernel_prob, self.pool_kernel)
        stride_size = self.sample_result(self.pool_stride_prob, self.pool_stride)
        padding = self.sample_result(self.pool_padding_prob, self.pool_padding)
        
        pool_type = self.sample_result(self.pool_type_prob, [0,1])
        if pool_type == 0:
            pool = MaxPool()
        else:
            pool = AvgPool()
        pool.create(no=_no, in_size=_in_size, kernel_sizde=[kernel_sizde,kernel_sizde], stride_size=[stride_size,stride_size], padding=padding)
        return pool
    
    def init_full(self, _no, _in_size):
        out_size = self.sample_result(self.full_size_prob, self.full_size)
        dropout = self.sample_result(self.full_dropout_prob, self.full_dropout)
        full = FullConnLayer()
        full.create(no=_no, in_size=_in_size, out_size=out_size, dropout=dropout)
        return full
        
        
    def sample_result(self, prob, options):
        
        
        if prob == 0.0:
            prob_list = []
            for _ in range(len(options)):
                prob_list.append(1/len(options))
        else:
            prob_list = prob
            
        r = np.random.choice(options, p=prob_list)
        return r
        
        
    def __str__(self): 
        _str = []
        s1 = 'id:%s,depth:%d,conv:%d,pool:%d,full:%d,uuid:%s'%(self.id, self.network_depth, self.conv_unit_size, self.pool_unit_size, self.full_unit_size, self.uuid())
        _str.append(s1)
        for u in self.units:
            _str.append(str(u))
        return '\n'.join(_str)
    
    def uuid(self):
        _str = []
        _sub_str = []
        for u in self.units:
            _sub_str.append(str(u))

        _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_= _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key#, _final_str_
        

    #net.create()
                    
        
        
        
        
        