import copy, random, math, os, queue

from algs.large_scale.genetic.statusupdatetool import StatusUpdateTool
from comm.log import Log


class ArcText():
    class unit():
        def __init__(self):
            self.Add_to = []
            self.Cat_to = []
            self.Type = 'None'
            self.Id = -1

        def __str__(self):
            return '%d,%s,cat_to:%s' % (self.Id, self.Type, '-'.join([str(i.Id) for i in self.Cat_to]))

    class unit_conv(unit):
        def __init__(self, kernel, stride, padding, in_size, out_size):
            super().__init__()
            self.Type = 'CONV'
            self.kernel = kernel
            self.stride = stride
            self.padding = padding
            self.in_size = in_size
            self.out_size = out_size

        def __str__(self):
            return super().__str__() + ',in:%d-%d-%d,out:%d-%d-%d,kernel:%d-%d,stride:%d,padding:%d-%d' % (
                self.in_size[0], self.in_size[1], self.in_size[2], self.out_size[0], self.out_size[1], self.out_size[2],
                self.kernel[0], self.kernel[1], self.stride, self.padding[0], self.padding[1])

    class unit_global_pool(unit):
        def __init__(self):
            super().__init__()
            self.Type = 'GLOBAL_POOL'

    class unit_bn_relu(unit):
        def __init__(self, in_channel):
            super().__init__()
            self.Type = 'BN_RELU'
            self.in_channel = in_channel

        def __str__(self):
            return super().__str__() + ',in_channel:%d' % (self.in_channel)

    class unit_interpolate(unit):
        def __init__(self, out_size):
            super().__init__()
            self.Type = 'INTERPOLATE'
            self.out_size = out_size

    class unit_fc(unit):
        def __init__(self, in_size, out_size):
            super().__init__()
            self.in_size = in_size
            self.out_size = out_size
            self.Type = 'FC'

        def __str__(self):
            return super().__str__() + ',in:%d,out:%d' % (self.in_size, self.out_size)

    def __init__(self):
        self.unitlist = []

    def __str__(self):
        ret = []
        for unit in self.unitlist:
            ret.append(unit.__str__())
        return '\n'.join(ret)

    def calculate_id(self):
        for i, unit in enumerate(self.unitlist):
            unit.Id = i

    def transform(self, Indi):
        W = StatusUpdateTool.get_input_weight()
        H = StatusUpdateTool.get_input_height()
        C = StatusUpdateTool.get_input_channel()
        Indi.calculate_flow()
        self.unitlist = [self.unit_interpolate(out_size=(W, H, C))]  # avoid err
        skip_layer = {}
        inqueue = []
        last = self.unitlist[-1]

        def connect_skip_layer(self, inqueue, last):
            if len(inqueue) == 0: return
            for i in inqueue:
                self.unitlist.append(self.unit_interpolate(out_size=(W, H, C)))
                i.Cat_to.append(self.unitlist[-1])
                self.unitlist[-1].Cat_to.append(last)
            inqueue.clear()

        for i, vertex in enumerate(Indi.vertices):
            if i in skip_layer: inqueue.extend(skip_layer[i])
            if i > 0:
                if vertex.type == 'bn_relu':
                    layer = self.unit_bn_relu(C)
                    last.Cat_to.append(layer)
                    last = layer
                    connect_skip_layer(self, inqueue, last)
                    self.unitlist.append(layer)
                if vertex.type == 'Global Pooling':
                    layer = self.unit_global_pool()
                    last.Cat_to.append(layer)
                    last = layer
                    connect_skip_layer(self, inqueue, last)
                    self.unitlist.append(layer)
                    # global pool is the final vertex in the graph
                    break
            for j, edge in enumerate(vertex.edges_out):
                if edge.to_vertex == Indi.vertices[i + 1]:
                    if edge.type == 'conv':
                        stride = int(pow(2, round(edge.stride_scale)))
                        layer = self.unit_conv(
                            in_size=(W, H, edge.input_channel),
                            out_size=(W // stride, H // stride, edge.output_channel),
                            kernel=(round(edge.filter_half_width) * 2 + 1, round(edge.filter_half_height) * 2 + 1),
                            stride=stride,
                            padding=(round(edge.filter_half_width), round(edge.filter_half_height)))
                        last.Cat_to.append(layer)
                        last = layer
                        W = layer.out_size[0]
                        H = layer.out_size[1]
                        C = layer.out_size[2]
                        connect_skip_layer(self, inqueue, last)
                        self.unitlist.append(layer)
                else:
                    pos = Indi.vertices.index(edge.to_vertex)
                    if pos not in skip_layer: skip_layer[pos] = []
                    skip_layer[pos].append(self.unitlist[-1])
        self.unitlist.append(self.unit_fc(in_size=C, out_size=StatusUpdateTool.get_num_class()))
        last.Cat_to.append(self.unitlist[-1])
        self.calculate_id()

    def to_pytorch_file(self):
        _ret = []
        _path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'model_template.py')
        file_handler = open(_path, 'r')
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# BEGIN':
                break
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# INIT':
                break
            _ret.append(_tmp[: -1])
        _str = []
        for unit in self.unitlist:
            if unit.Type == 'CONV':
                _str.append(
                    'self.unit_%d = torch.nn.Conv2d(%d, %d, kernel_size=(%d, %d), stride=%d, padding=(%d, %d))' % (
                        unit.Id, unit.in_size[2],
                        unit.out_size[2],
                        unit.kernel[0],
                        unit.kernel[1],
                        unit.stride,
                        unit.padding[0],
                        unit.padding[1]))
            if unit.Type == 'FC':
                _str.append('self.unit_%d = torch.nn.Linear(%d, %d)' % (unit.Id, unit.in_size, unit.out_size))
            if unit.Type == 'BN_RELU':
                _str.append(
                    'self.unit_%d = torch.nn.Sequential(torch.nn.BatchNorm2d(%d), torch.nn.ReLU(inplace=True))' % (
                        unit.Id, unit.in_channel))
        _ret.extend([' ' * 8 + i for i in _str])
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# FORWARD':
                break
            # 去除回车
            _ret.append(_tmp[: -1])
        _str = []
        _str.append('x = {}')
        _str.append('x[0] = input')
        for unit in self.unitlist: unit.visit = False
        for unit in self.unitlist:
            if unit.Type == 'CONV':
                _str.append('x[%d] = self.unit_%d(x[%d])' % (unit.Id, unit.Id, unit.Id))
            if unit.Type == 'INTERPOLATE':
                _str.append('x[%d] = torch.nn.functional.interpolate(x[%d], size = (%d, %d))' % (
                    unit.Id, unit.Id, unit.out_size[0], unit.out_size[1]))
            if unit.Type == 'BN_RELU':
                _str.append('x[%d] = self.unit_%d(x[%d])' % (unit.Id, unit.Id, unit.Id))
            if unit.Type == 'GLOBAL_POOL':
                _str.append('x[%d] = torch.squeeze(torch.squeeze(torch.nn.AdaptiveAvgPool2d((1, 1))(x[%d]), 3), 2)' % (
                    unit.Id, unit.Id))
            if unit.Type == 'FC':
                _str.append('x[%d] = self.unit_%d(x[%d])' % (unit.Id, unit.Id, unit.Id))
            for to_vertex in unit.Cat_to:
                if to_vertex.visit == False:
                    _str.append('x[%d] = x[%d]' % (to_vertex.Id, unit.Id))
                else:
                    _str.append('torch.cat([x[%d], x[%d]], 1)' % (to_vertex.Id, unit.Id))
                to_vertex.visit = True
        _str.append('return x[%d]' % (len(self.unitlist) - 1))
        _ret.extend([' ' * 8 + i for i in _str])
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# END':
                break
            # 去除回车
            _ret.append(_tmp[: -1])
        _ret = '\n'.join(_ret)
        return _ret


class Vertex(object):
    '''
    edges_in, edges_out, HasField(bn_relu/linear), 
    inputs_mutable, outputs_mutable, properties_mutable
    '''

    def __init__(self,
                 edges_in,
                 edges_out,
                 type='linear',
                 inputs_mutable=1,
                 outputs_mutable=1,
                 properties_mutable=1):
        '''
        edges_in / edges_out : 使用set 
        each vertex can be inlear / 1*relu + 1*bn
        '''
        self.edges_in = edges_in
        self.edges_out = edges_out
        self.type = type  # ['linear' / 'bn_relu']

        self.inputs_mutable = inputs_mutable
        self.outputs_mutable = outputs_mutable
        self.properties_mutable = properties_mutable

        self.input_channel = 0
        # Each vertex represents a 2ˆs x 2ˆs x d block of nodes. s and d are positive
        # integers computed dynamically from the in-edges. s stands for "scale" so
        # that 2ˆx x 2ˆs is the spatial size of the activations. d stands for "depth",
        # the number of channels.


class Edge(object):
    '''
    No Need:type, depth_factor, filter_half_width, filter_half_height, 
            stride_scale, depth_precedence, scale_precedence
    '''

    def __init__(self,
                 from_vertex,
                 to_vertex,
                 type='identity',
                 depth_factor=1,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=0):
        self.from_vertex = from_vertex  # Source vertex ID.
        self.to_vertex = to_vertex  # Destination vertex ID.
        self.type = type

        # In this case, the edge represents a convolution.
        # 控制 channel 大小, this.channel = last channel * depth_factor
        self.depth_factor = depth_factor
        # Controls the strides(步幅) of the convolution. It will be 2ˆstride_scale. WHY ?????
        self.stride_scale = stride_scale

        if type == 'conv':
            # 卷积核 size
            # filter_width = 2 * filter_half_width + 1.
            self.filter_half_width = filter_half_width
            self.filter_half_height = filter_half_height
            # 定义卷积步长, 卷积步长必须是 2 的幂次方？

        # determine the inputs takes precedence in deciding the resolved depth or scale.
        # self.depth_precedence = edge_proto.depth_precedence
        # self.scale_precedence = edge_proto.scale_precedence

        self.input_channel = 0
        self.output_channel = 0


class Individual():
    input_size_height = StatusUpdateTool.get_input_height()
    input_size_width = StatusUpdateTool.get_input_weight()
    input_size_channel = StatusUpdateTool.get_input_channel()

    output_size_height = 1
    output_size_width = 1
    output_size_channel = StatusUpdateTool.get_num_class()

    def __init__(self, individual_id, learning_rate=0.05):
        '''
        注意，vertice 和 edges 中应该存引用
        '''
        self.fitness = -1.0
        self.learning_rate = learning_rate
        self.individual_id = individual_id
        # input layer
        l0 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='linear',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        # Global Pooling layer
        l1 = Vertex(edges_in=set(),
                    edges_out=set(),
                    type='Global Pooling',
                    inputs_mutable=0,
                    outputs_mutable=0,
                    properties_mutable=0)
        self.vertices = []
        self.vertices.append(l0)
        self.vertices.append(l1)

        # edge
        edg1 = Edge(from_vertex=l0, to_vertex=l1, type='identity')

        edg1.input_channel = self.input_size_channel
        edg1.output_channel = self.input_size_channel
        self.edges = []
        self.edges.append(edg1)

        l0.edges_out.add(edg1)
        l1.edges_in.add(edg1)
        # self.add_vertex(1, 'linear', 'identity')
        # self.calculate_flow()

    def add_edge(self,
                 from_vertex_id,
                 to_vertex_id,
                 edge_type='identity',
                 depth_factor=1,
                 filter_half_width=None,
                 filter_half_height=None,
                 stride_scale=0):
        """
        Adds an edge to the DNA graph, ensuring internal consistency.
        """
        edge = Edge(from_vertex=self.vertices[from_vertex_id],
                    to_vertex=self.vertices[to_vertex_id],
                    type=edge_type,
                    depth_factor=depth_factor,
                    filter_half_width=filter_half_width,
                    filter_half_height=filter_half_height,
                    stride_scale=stride_scale)
        self.edges.append(edge)
        self.vertices[from_vertex_id].edges_out.add(edge)
        self.vertices[to_vertex_id].edges_in.add(edge)
        return edge

    def calculate_flow(self):
        '''
        按顺序计算神经网络每层的输入输出参数
        '''
        self.vertices[0].input_channel = self.input_size_channel
        # self.vertices[0].output_channel = self.input_size_channel
        # self.vertices[-1].input_channel = self.output_size_channel
        # self.vertices[-1].output_channel = self.output_size_channel

        for i, vertex in enumerate(self.vertices[1:], start=1):
            vertex.input_channel = 0
            Log.debug('vertex [%d].%d' % (i, vertex.input_channel))
            # print('vertex [', i, '].{}'.format(vertex.input_channel), end=' ')
            for edge in vertex.edges_in:
                edge.input_channel = edge.from_vertex.input_channel
                edge.output_channel = int(
                    edge.input_channel * edge.depth_factor)
                vertex.input_channel += edge.output_channel

                f_ver = self.vertices.index(edge.from_vertex)
                if edge.type == 'identity':
                    f_h = 'N'
                else:
                    f_h = edge.filter_half_height
                if edge.type == 'identity':
                    f_w = 'N'
                else:
                    f_w = edge.filter_half_width
                Log.debug(', %s.%s_s%s,%s,%s' % (f_ver, edge.type[0], edge.stride_scale, f_h, f_w))
                # print(', {}.{}_s{},{},{}'.format(f_ver, edge.type[0], edge.stride_scale, f_h, f_w), end=' ')
        Log.debug('[calculate_flow] finish')
        # print('[calculate_flow] finish')

    def remove_edge(self, edge):
        edge.from_vertex.edges_out.remove(edge)
        edge.to_vertex.edges_in.remove(edge)
        self.edges.remove(edge)
        del edge

    def remove_vertex(self, vertex_id):
        edge = None
        for i, edge in enumerate(self.edges):
            if edge.from_vertex == self.vertices[vertex_id] and \
                    edge.to_vertex == self.vertices[vertex_id + 1]:
                changed_edge = self.edges[i]
                self.vertices[vertex_id].edges_out.remove(changed_edge)
                break
        for i in list(self.vertices[vertex_id].edges_in):
            self.remove_edge(i)
        for i in list(self.vertices[vertex_id].edges_out):
            self.remove_edge(i)
        self.vertices[vertex_id - 1].edges_out.add(changed_edge)
        changed_edge.from_vertex = self.vertices[vertex_id - 1]
        self.vertices.remove(self.vertices[vertex_id])

    def add_vertex(self, after_vertex_id, vertex_type='linear', edge_type='identity'):
        '''
        3.0: 所有 vertex 和 edg 中记录的都是引用
        '''
        changed_edge = None
        # 先寻找那条应该被移除的边, 将其删除
        # for i in self.vertices[after_vertex_id - 1].edges_out:
        #    if i.to_vertex == self.vertices[after_vertex_id]:
        #        self.vertices[after_vertex_id - 1].edges_out.remove(i)
        #        break
        # for i in self.vertices[after_vertex_id].edges_in:
        #    if i.from_vertex == self.vertices[after_vertex_id - 1]:
        #        self.vertices[after_vertex_id].edges_in.remove(i)
        #        break
        for i, edge in enumerate(self.edges):
            if edge.from_vertex == self.vertices[
                after_vertex_id - 1] and edge.to_vertex == self.vertices[after_vertex_id]:
                changed_edge = self.edges[i]
        self.vertices[after_vertex_id - 1].edges_out.remove(changed_edge)
        self.vertices[after_vertex_id].edges_in.remove(changed_edge)

        # 创建新的 vertex, 并加入队列
        vertex_add = Vertex(edges_in=set(), edges_out=set(), type=vertex_type)
        self.vertices.insert(after_vertex_id, vertex_add)

        # 创建新的 edge, 并加入队列
        if edge_type == 'conv':
            depth_f = random.uniform(0.5, 2)
            filter_h = 1
            filter_w = 1
            stride_s = math.floor(random.random() * 2)
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='conv',
                             depth_factor=depth_f,
                             filter_half_height=filter_h,
                             filter_half_width=filter_w,
                             stride_scale=0)
        else:
            edge_add1 = Edge(from_vertex=self.vertices[after_vertex_id - 1],
                             to_vertex=self.vertices[after_vertex_id],
                             type='identity')
        # 取代的那条边后移
        changed_edge.from_vertex = self.vertices[after_vertex_id]
        # edge_add2 = Edge(from_vertex=self.vertices[after_vertex_id],to_vertex=self.vertices[after_vertex_id + 1])
        self.edges.append(edge_add1)
        # self.edges.append(edge_add2)

        self.vertices[after_vertex_id - 1].edges_out.add(edge_add1)
        vertex_add.edges_in.add(
            edge_add1), vertex_add.edges_out.add(changed_edge)
        self.vertices[after_vertex_id + 1].edges_in.add(changed_edge)

    def has_edge(self, from_vertex_id, to_vertex_id):
        vertex_before = self.vertices[from_vertex_id]
        vertex_after = self.vertices[to_vertex_id]
        for edg in self.edges:
            if edg.from_vertex == vertex_before and edg.to_vertex == vertex_after:
                return True
        return False

    def to_pytorch_file(self):
        _ret = []
        _path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'cifar10.py')
        file_handler = open(_path, 'r')
        # 忽略BEGIN之前的内容
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# BEGIN':
                break
            # 去除回车
            _ret.append(_tmp[: -1])
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# INIT':
                break
            # 去除回车
            _ret.append(_tmp[: -1])
        _str = []
        _str.append('self.layer_vertex = torch.nn.ModuleList()')
        # self.layer_vertex = torch.nn.ModuleList()
        # print('init vertex', end='')
        for i, vertex in enumerate(self.vertices):
            # print('v{} '.format(i), end='')
            # 默认第一层和最后一层 vertex 非 hidden 层
            if vertex.type == 'bn_relu':
                _str.append(
                    'self.layer_vertex.append(torch.nn.Sequential(torch.nn.BatchNorm2d(%d), torch.nn.ReLU(inplace=True)))' % (
                        vertex.input_channel))
                # self.layer_vertex.append(
                #    torch.nn.Sequential(torch.nn.BatchNorm2d(vertex.input_channel),
                #                        torch.nn.ReLU(inplace=True)))
            elif vertex.type == 'Global Pooling':
                _str.append('self.layer_vertex.append(torch.nn.Sequential(torch.nn.Linear(%d, %d)))' % (
                    vertex.input_channel, self.output_size_channel))
                # self.layer_vertex.append(
                #    torch.nn.Sequential(
                #        torch.nn.Linear(vertex.input_channel, DNA.output_size_channel)))
            else:
                _str.append('self.layer_vertex.append(None)')
                # self.layer_vertex.append(None)
        _str.append('self.layer_edge = torch.nn.ModuleList()')
        # self.layer_edge = torch.nn.ModuleList()
        # print('\ninit edges', end='')
        for i, edge in enumerate(self.edges):
            # print('e{}:'.format(i), end='')
            if edge.type == 'conv':
                # print('{},{},{}'.format(edge.filter_half_height, edge.filter_half_width,
                #                        edge.stride_scale),
                #      end=' ')
                _str.append(
                    'self.layer_edge.append(torch.nn.Conv2d(%d, %d, kernel_size=(%d, %d), stride=%d, padding=(%d, %d)))' % (
                        edge.input_channel,
                        edge.output_channel,
                        round(edge.filter_half_height) * 2 + 1,
                        round(edge.filter_half_width) * 2 + 1,
                        pow(2, round(edge.stride_scale)),
                        round(edge.filter_half_height),
                        round(edge.filter_half_width)))
            else:
                # print(end=' ')
                _str.append('self.layer_edge.append(None)')
                # self.layer_edge.append(None)
        # print('')
        # self.batch_size = Evolution_pop.BATCH_SIZE
        _ret.extend([' ' * 8 + i for i in _str])
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# FORWARD':
                break
            _ret.append(_tmp[: -1])
        _str = []
        _str.append('block_h = input.shape[0]')
        # block_h = input.shape[0]
        _str.append('x = { 0: input }')
        # x = {
        #    0: input,
        # }
        for index, vert in enumerate(self.vertices[1:], start=1):
            # length = len(x)
            # a = torch.empty(block_h, 0, 0, 0)
            # NOTE: 不会出现一个点没有前向边
            # _str.append('a = torch.empty(block_h, 0, 0, 0)')
            _str.append('#process vertex: %d' % (index))
            for j, edg in enumerate(vert.edges_in):
                ind_edg = self.edges.index(edg)
                ind_x = self.vertices.index(edg.from_vertex)
                _str.append('t = x[%d]' % (ind_x))
                # t = x[ind_x]
                if edg.type == 'conv':
                    _str.append('t = self.layer_edge[%d](x[%d])' % (ind_edg, ind_x))
                    # t = self.layer_edge[ind_edg](x[ind_x])
                if j == 0:
                    _str.append('a = torch.empty(block_h, 0, t.shape[2], t.shape[3])')
                    # a = torch.empty(block_h, 0, t.shape[2], t.shape[3])
                _str.append('if t.shape[2:] != a.shape[2:]: t = F.interpolate(t, size = a.shape[2:])')
                # if t.shape[2:] != a.shape[2:]:
                #    #差值调整大小
                #    t = F.interpolate(t, size = a.shape[2:])
                _str.append('a = torch.cat((a, t), dim=1)')
                # a = torch.cat((a, t), dim=1)
            if self.vertices[index].type == 'linear':
                _str.append('x[%d] = a' % (index))
                # x[index] = a
            elif self.vertices[index].type == 'bn_relu':
                _str.append('x[%d] = self.layer_vertex[%d](a)' % (index, index))
                # x[index] = layer_vert(a)
            elif self.vertices[index].type == 'Global Pooling':
                _str.append('a = torch.squeeze(torch.squeeze(torch.nn.AdaptiveAvgPool2d((1, 1))(a), 3), 2)')
                _str.append('x[%d] = self.layer_vertex[%d](a)' % (index, index))
                # x[index] = layer_vert(a)
        _str.append('return x[len(x) - 1]')
        # return x[len(x) - 1]
        _ret.extend([' ' * 8 + i for i in _str])
        while True:
            _tmp = file_handler.readline()
            if _tmp.strip() == '# END':
                break
            _ret.append(_tmp[: -1])
        file_handler.close()
        _ret = '\n'.join(_ret)
        Log.debug('generate success...')
        # print('generate success...')
        return _ret


class Population():
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d%05d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(individual_id=indi_no)
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        # TODO: 变异产生后代
        self.number_id = 0
        self.individuals = []
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d%05d' % (self.gen_no, self.number_id)
            indi.individual_id = indi_no
            self.number_id += 1
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)
