import re
import os
from train.utils import TrainConfig
import algs.performance_pred.template.model_template as template
from compute.file import get_algo_local_dir, get_population_dir

class GenPyModel(object):
    def __init__(self):
        self.id_pat=re.compile('id:(\w+),depth:(\d+),conv:(\d+),pool:(\d+),full:(\d+)')
        # id_pat=re.compile('id:(\w+),')
        self.conv_pat=re.compile('(\d+),CONV,in:([\d-]+),out:([\d-]+),kernel:([\d-]+),stride:([\d-]+),groups:(\d+),padding:(\d+),add_to:([-\dNone]+),cat_to:([-\dNone]+)')
        self.avgp_pat=re.compile('(\d+),AVG_POOL,in:([\d-]+),out:([\d-]+),kernel:([\d-]+),stride:([\d-]+),padding:(\d+)')
        self.maxp_pat=re.compile('(\d+),MAX_POOL,in:([\d-]+),out:([\d-]+),kernel:([\d-]+),stride:([\d-]+),padding:(\d+)')
        self.full_pat=re.compile('(\d+),FULL,in:(\d+),out:(\d+),dropout:([\d.]+)')
        self.layer_types=['conv','avgp','maxp','full']
        
        self.file_string=template.__doc__
        self.save_dir='convert_models'
        tab_num=2
        tab_width=4
        self.space_str=' '*tab_num*tab_width

        # suppress the max channels of the dense path
        self.max_cat=1e100
        # the number of the classes
        self.out_class_num= TrainConfig.get_out_cls_num()

    def _parse_txt(self,networks_file):
        """
        Args:
            networks_file       (str): the path of the network file
        Return: 
            inter_model_list    (list): the list contains intermediate model's representation
        """
        # read the intermediate codes...
        with open(networks_file,'r') as f:
            # remove '\n'
            code_lines=[_.replace('\n','') for _ in f.readlines()]

        # parse the string lines
        code_block_list=[]
        cur_block=[]
        for l in code_lines:
            if l.startswith('==='):
                code_block_list+=[cur_block]
                cur_block=[]
            else:
                cur_block+=[l]
        # code_block_l+=[cur_block]

        inter_model_list=[]
        for block in code_block_list:
            model={}
            layers=[]
            model['layers']=layers
            for i,k in enumerate(block[0].split(',')):
                key=k.split(':')[0]
                val=k.split(':')[1]
                model[key]=int(val) if val.isdigit() else val

            for l in block[1:]:
                
                for i,pat in enumerate([self.conv_pat,self.avgp_pat,self.maxp_pat,self.full_pat]):
                    mat_res=pat.match(l)
                    if mat_res is not None:
                        break
                assert mat_res 
                mat_groups=mat_res.groups()
            
                # it's the layer
                layer_type=self.layer_types[i]
                layer={}
                layer['type']=layer_type
                for i,k in enumerate(['N:']+ [_  for _ in pat.pattern.split(',') if _.find(':')!=-1 ]):
                    key=k.split(':')[0]
                    val=mat_groups[i]
                    if val.find('-') != -1:
                        val=val.split('-')
                        val=[int(_)for _ in val]
                    elif val=='None':
                        val=None
                    elif val.find('.')!=-1:
                        val=float(val)
                    elif val.isdigit():
                        val=int(val)
                    else:
                        assert 0,val
                        
                    layer[key]=val
                layers+=[layer]
            inter_model_list+=[model]
        return inter_model_list

    def _gen_py_str(self,iml):
        """
        Args:
            iml             (list): inter model list
        Return:
            py_str_models   (list): the string list contain py codes and forward codes
        """
        space_str = self.space_str
        out_class_num = self.out_class_num
        py_str_models=[]
        for imodel in iml:
            layers=imodel['layers']
            # shortcut path list
            scp_list=[[] for _ in range(len(layers))]
            # dense path list
            dsp_list=[[] for _ in range(len(layers))]
            # store the dense path concatenated channels
            dpc_list=[0]*len(layers)
            # store the channels before i-th layer
            before_i=[0]*len(layers)
            before_i[0]=3 # for channel
            channel_track=[ [] for _ in range(len(layers))]
            model_init=[]
            model_forward=[]
            for i,l in enumerate(layers):
                layer_init=None
                layer_forward=None
                l_type=l['type']
                if l_type=='conv':
                    # 1. decode some parameters for convolution layer.
                    pos=l['N']
                    ih,iw,ic=l['in']
                    oh,ow,oc=l['out']
                    kernel=l['kernel']
                    stride=l['stride']
                    groups=l['groups']
                    padding=l['padding']
                    
                    # check for insures 
                    assert i==pos,'are you kidding me?'
                    assert ih==iw,'image height and width should be the same.'
                    assert (ih+2*padding-kernel[0])//stride[0]+1==oh,'padding setting is invalid.'
                    
                    if ic%groups!=0 or oc % groups!=0:
                        print("WRANING: input or output channel can not be divided by groups, change to default groups")
                        groups=1
                        
                    add_to=l['add_to']
                    if isinstance(add_to,int):
                        add_to=[add_to]
                    cat_to=l['cat_to']
                    if isinstance(cat_to,int):
                        cat_to=[cat_to]
                    
                    # 2. generate `py` codes
                    # 2.1 regular 
                    ic=ic+dpc_list[i]
                    # check
                    if len(dsp_list[i])!=0 and  sum([before_i[_+1] for _ in dsp_list[i] ]) != dpc_list[i]:
                        print(1,i)
                    if ic>self.max_cat and dpc_list[i] > 0: 
                        # if i==51:
                        #     print(i)
                        # use suppression strategy
                        # dpnum=len(dsp_list[i])+1
                        adjust_rate=self.max_cat/ic 
                        adjust_cout=0
                        layer_init=''
                        ic_another_way=0
                        for fpos in dsp_list[i]:
                            # fpos_org_c=dpc_list[fpos+1]+layers[fpos]['out'][2]
                            fpos_org_c=before_i[fpos+1]
                            ic_another_way+=fpos_org_c
                            if fpos_org_c>self.max_cat:
                                print('fpos_org_c > max_cat ->',fpos_org_c)
                            adjust_num=int(fpos_org_c*adjust_rate)
                            layer_init+='\n%sself.adjust%d_%d = nn.Conv2d(%d, %d, (1, 1), (1, 1))'%(space_str,fpos,pos,fpos_org_c,adjust_num)
                            adjust_cout+=adjust_num
                        org_in_c=l['in'][2]
                        ic_another_way+=org_in_c
                        adjust_in_c=int(org_in_c*adjust_rate)
                        layer_init+='\n%sself.adjust%d = nn.Conv2d(%d, %d, (1, 1), (1, 1))'%(space_str,pos,org_in_c,adjust_in_c)
                        adjust_cout+=adjust_in_c
                        
                        if adjust_cout > self.max_cat:
                            print('adjust_cout > max_cat ->', adjust_cout)
                        if adjust_cout % groups !=0:
                            groups=1
                        layer_init+='\n%sself.conv%d = nn.Sequential(nn.Conv2d(%d, %d, (%d, %d), (%d, %d), padding=%d, groups=%d, bias=False), nn.BatchNorm2d(%d))' \
                        %(space_str,pos,adjust_cout,oc,kernel[0],kernel[1],stride[0],stride[1],padding,groups,oc)
                        
                        before_i[i]=adjust_cout
                    else:
                        layer_init='\n%sself.conv%d = nn.Sequential(nn.Conv2d(%d, %d, (%d, %d), (%d, %d), padding=%d, groups=%d, bias=False), nn.BatchNorm2d(%d))' \
                        %(space_str,pos,ic,oc,kernel[0],kernel[1],stride[0],stride[1],padding,groups,oc)
                        before_i[i]=ic
                    # 2.2 auxiliary codes for shortcut path
                    #    to make sure the element-wise adding 
                    #    is valid by adding a new convolution layer.
                    # if add_to is not None:
                    #     for tpos in add_to:  
                    #         tl=layers[tpos]
                    #         tih,tiw,tic=tl['in']
                    #         if oc != tic:
                    #             layer_init+='\n%sself.side_%d_%d=nn.Conv2d(%d,%d,(1,1),(1,1))'%(space_str,pos,tpos,oc,tic)
                    #         scp_list[tpos]+=[pos]
                    # already generated in 3.1
                            
                    # 3. generate some `forward` codes
                    # 3.1 consider the short path
                    # layer_forward='\n%sx%d=self.conv%d(x%d)'%(space_str,pos-1,pos,pos-1)
                    layer_forward=''
                    for fpos in scp_list[pos]:
            
                        # total_c=dpc_list[fpos+1]+layers[fpos]['out'][2]
                        total_c=before_i[fpos+1]
                        front_out_c=layers[pos-1]['out'][2]
                        
                        if total_c!=front_out_c:
                            # use auxiliary conv
                            layer_init+='\n%sself.side_%d_%d = nn.Conv2d(%d, %d, (1, 1), (1, 1))'%(space_str,fpos,pos,total_c,front_out_c)
                            layer_forward+='\n%sx%d = self.side_%d_%d(x%d) + x%d'%(space_str,pos-1,fpos,pos,fpos,pos-1)
                        else:
                            layer_forward+='\n%sx%d = x%d + x%d'%(space_str,pos-1,pos-1,fpos)
                    if i!=0 and layers[i-1]['type']=='conv':
                        layer_forward+='\n%sx%d = F.relu(x%d, inplace=True)'% (space_str,pos-1,pos-1)
                    # 3.2 consider the dense path 
                    if ic>self.max_cat and dpc_list[i] > 0:
                        layer_forward+='\n%sx%d = self.adjust%d(x%d)' % (space_str,pos-1, pos, pos-1)

                    for fpos in dsp_list[pos]:
                        if ic>self.max_cat and dpc_list[i] > 0:
                            # adjust the channel
                            layer_forward+='\n%sx%d = torch.cat([x%d, self.adjust%d_%d(x%d)], dim=1)' % (space_str,pos-1, pos-1,fpos, pos, fpos)
                        else:
                            layer_forward+='\n%sx%d = torch.cat([x%d, x%d], dim=1)' %(space_str,pos-1,pos-1,fpos)
                    if i==0:
                        layer_forward+='\n%sx%d = self.conv%d(x)'%(space_str,i,i)
                    else:
                        layer_forward+='\n%sx%d = self.conv%d(x%d)'%(space_str,i,i,i-1)
                        
                    # 4. update shortcut list
                    if add_to is not None:
                        for tpos in add_to:  
        #                     tl=layers[tpos]
                            scp_list[tpos]+=[pos]
                    
                    # 5. update dense path list
                    if cat_to is not None:
                        # forward calc
                        temp_oc=oc + sum([before_i[_+1] for _ in dsp_list[pos+1]])
                        if temp_oc > self.max_cat:
                            adjust_rate=self.max_cat/temp_oc
                            adjust_cout=0
                            for fpos in dsp_list[pos+1]:
                                fpos_org_c=before_i[fpos+1]
                                adjust_num=int(fpos_org_c*adjust_rate)
                                adjust_cout+=adjust_num
                            adjust_cout+=int(oc*adjust_rate)
                        for tpos in cat_to:
                           
                            # if ic>self.max_cat and dpc_list[i] > 0:
                            #     dpc_list[tpos]+=adjust_cout
                            #     channel_track[tpos]+=[adjust_cout]
                            # else:
                            #     dpc_list[tpos]+=oc+dpc_list[pos+1]
                            #     channel_track[tpos]+=[oc+dpc_list[pos+1]]
                            if temp_oc > self.max_cat:
                                dpc_list[tpos]+=adjust_cout
                                channel_track[tpos]+=[adjust_cout]
                            else:
                                dpc_list[tpos]+=oc+dpc_list[pos+1]
                                channel_track[tpos]+=[oc+dpc_list[pos+1]]
                            dsp_list[tpos]+=[pos]
                elif l_type=='avgp' or l_type=='maxp':
                    # 1. get params
                    pos=l['N']
                    ih,iw,ic=l['in']
                    oh,ow,oc=l['out']
                    kernel=l['kernel']
                    stride=l['stride']
                    padding=l['padding']
                    # check for insures 
                    assert i==pos,'are you kidding me?'
                    assert ih==iw,'image height and width should be the same.'
                    assert (ih+2*padding-kernel[0])//stride[0]+1==oh,'padding setting is invalid.'
                    
                    # 2. generate init code
                    
                    layer_init='\n%sself.%s%d = nn.%s((%d, %d), (%d, %d), padding=%d)' \
                    %(space_str,l_type,pos,'AvgPool2d' if l_type=='avgp' else 'MaxPool2d',kernel[0],kernel[1],stride[0],stride[1],padding)
                    
                    # 3. generate forward codes
                    layer_forward=''
                    if i!=0 and layers[i-1]['type']=='conv':
                        layer_forward+='\n%sx%d = F.relu(x%d)'%(space_str,pos-1,pos-1)
                    
                    if i==0:
                        layer_forward+='\n%sx%d = self.%s%d(x)'%(space_str,pos,l_type,pos)
                    else:
                        layer_forward+='\n%sx%d = self.%s%d(x%d)'%(space_str,pos,l_type,pos,pos-1)
                    
                
                elif l_type=='full':
                    # 1. read params
                    pos=l['N']
                    feat_in=l['in']
                    feat_out=l['out']
                    dropout=l['dropout']
                    
                    # 2. generate init code
                    layer_init='\n%sself.linear%d = nn.Linear(%d, %d)'%(space_str,pos,feat_in,feat_out)
                    layer_init+='\n%sself.dropout%d = nn.Dropout(%.2f)'%(space_str,pos,dropout)
                    
                    # 3. forward
                    layer_forward=''
                    last_layer_type=layers[pos-1]['type']
                    if last_layer_type=='conv' or last_layer_type=='maxp' or last_layer_type=='avgp':
                        layer_forward+='\n%sx%d = x%d.view(len(x%d),-1)'%(space_str,pos-1,pos-1,pos-1)
                    
                    if i==0:
                        layer_forward+='\n%sx%d = F.relu(self.linear%d(x), inplace=True)'%(space_str,pos,pos)
                    else:
                        layer_forward+='\n%sx%d = F.relu(self.linear%d(x%d), inplace=True)'%(space_str,pos,pos,pos-1)
                    layer_forward+='\n%sx%d = self.dropout%d(x%d)' %(space_str,pos,pos,pos)
                    
                else:
                    assert 0,l
                assert layer_init and layer_forward,'layer can not be None'
                model_init+=[layer_init]
                model_forward+=[layer_forward]
            
            # add the last fully connection layer.
            last_layer_type=l['type']
            if last_layer_type=='conv' or last_layer_type=='maxp' or last_layer_type=='avgp':
                # 1. read params
                pos=l['N']
                oh,ow,oc=l['out']
                feat_in=oh*ow*oc
                
                # 2. generate init code
                layer_init='\n%sself.linear = nn.Linear(%d, %d)'%(space_str,feat_in,out_class_num)
                
                # 3. forward
                layer_forward='\n%sout=x%d'%(space_str,pos)
            
            elif last_layer_type=='full':
                # 1. read params
                pos=l['N']
                feat_out=l['out']
                
                # 2. generate init code
                layer_init='\n%sself.linear = nn.Linear(%d, %d)'%(space_str,feat_out,out_class_num)
                
                # 3. forward
                layer_forward='\n%sout=x%d'%(space_str,pos)
            else:
                assert 0,l
            model_init+=[layer_init]
            model_forward+=[layer_forward]
            py_str_models+=[(model_init,model_forward)]
        return py_str_models

    def _write(self,py_str_models,inter_model_list):
        """
        Generate the executable python files
        Args:
            py_str_models       (list): python model strings in list
            inter_model_list    (list): the inter model list
        Return:
            None
        """
        save_dir=self.save_dir
        target_file_string=self.file_string
        for i,py_str_model in enumerate(py_str_models): 
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open('%s/%s.py'%(save_dir,inter_model_list[i]['id']),'w') as f:
                content=target_file_string.split('#ANCHOR-generated_init')
                middle=''
                for l in py_str_model[0]:
                    middle+=l
                content=content[0]+middle+content[1]
                
                content=content.split('#ANCHOR-generate_forward')
                middle=''
                # f.write('\n'+'-'*200)
                for l in py_str_model[1]:
                    middle+=l
                content=content[0]+middle+content[1]
                f.write(content)

    def convert(self, network_file=os.path.join(get_population_dir(),'networks.txt'), 
                      save_dir= os.path.join(get_algo_local_dir(), 'scripts'),
                       max_cat=512):
        """
        Convert the model description to real python model file
        Args:
            network_file    (str): the txt file contains the network's info
            save_dir        (str): generated python file's saving directory
            max_cat         (int): maximum channels of concatenating
        """
        out_class_num= TrainConfig.get_out_cls_num()
        self.save_dir=save_dir
        self.out_class_num=out_class_num
        self.max_cat=max_cat
        iml = self._parse_txt(network_file)
        psms=self._gen_py_str(iml)
        self._write(psms,iml)
        