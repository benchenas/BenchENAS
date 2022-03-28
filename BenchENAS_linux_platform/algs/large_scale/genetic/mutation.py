import random
class StructMutation():
    def mutate(self, dna):
        possible_mutations = [
            #'ALTER-LEARNING-RATE',
            'IDENTITY',
            'INSERT-CONVOLUTION',
            'REMOVE-CONVOLUTION',
            'ALTER-STRIDE',
            'ALTER-NUMBER-OF-CHANNELS',
            'FILTER-SIZE',
            'ADD-SKIP',
            'REMOVE-SKIP',
        ]
        while True:
            mutation = random.choice(possible_mutations)
            if mutation == 'ALTER-LEARNING-RATE' and self.mutate_learningRate(dna):
                break
            if mutation == 'IDENTITY':
                pass
            if mutation == 'INSERT-CONVOLUTION' and self.insert_convolution(dna):
                break
            if mutation == 'REMOVE-CONVOLUTION' and self.remove_convolution(dna):
                break
            if mutation == 'ALTER-STRIDE' and self.mutate_alter_stride(dna):
                break
            if mutation == 'ALTER-NUMBER-OF-CHANNELS' and self.mutate_alter_channels(dna):
                break
            if mutation == 'FILTER-SIZE' and self.mutate_filter_size(dna):
                break
            if mutation == 'ADD-SKIP' and self.mutate_add_skip(dna):
                break
            if mutation == 'REMOVE-SKIP' and self.mutate_remove_skip(dna):
                break
        return
    def mutate_alter_channels(self, dna):
        possible_edge = [edge for edge in dna.edges if edge.type == 'conv']
        if len(possible_edge) == 0:
            return False
        edge = random.choice(possible_edge)
        edge.depth_factor *= (2 ** random.uniform(-1.0, 1.0))
        return True
    
    def mutate_alter_stride(self, dna):
        possible_edge = [edge for edge in dna.edges if edge.type == 'conv']
        if len(possible_edge) == 0:
            return False
        edge = random.choice(possible_edge)
        edge.stride_scale = max(0, edge.stride_scale + random.uniform(-1, 1))
        return True
                
    def mutate_add_skip(self, dna):
        vertices = random.sample(range(len(dna.vertices)), k = 2)
        u = min(vertices)
        v = max(vertices)
        if dna.has_edge(u, v):
            return False
        dna.add_edge(u, v)
        return True
    
    def mutate_remove_skip(self, dna):
        possible_edge = []
        for edge in dna.edges:
            u = dna.vertices.index(edge.from_vertex)
            v = dna.vertices.index(edge.to_vertex)
            if abs(u - v) != 1:
                possible_edge.append(edge)
        if len(possible_edge) == 0:
            return False
        edge = random.choice(possible_edge)
        dna.remove_edge(edge)
        return True

    def mutate_filter_size(self, dna):
        possible_edge = [edge for edge in dna.edges if edge.type == 'conv']
        if len(possible_edge) == 0:
            return False
        edge = random.choice(possible_edge)
        if random.random() > 0.5:
            edge.filter_half_height *= (2 ** random.uniform(-1, 1))
        if random.random() > 0.5:
            edge.filter_half_width *= (2 ** random.uniform(-1, 1))
        return True

    def mutate_learningRate(self, dna):
        # mutated_dna = copy.deepcopy(dna)
        # Mutate the learning rate by a random factor between 0.5 and 2.0,
        # uniformly distributed in log scale.
        dna.learning_rate *= (2 ** random.uniform(-1.0, 1.0))
        return True

    def insert_convolution(self, dna):
        # 随机选择一个 vertex_id 插入 vertex
        after_vertex_id = random.choice(range(1, len(dna.vertices)))
        # TODO: how it supposed to mutate
        vertex_type = 'linear'
        if random.random() > 0.2:
            vertex_type = 'bn_relu'

        edge_type = 'identity'
        if random.random() > 0.2:
            edge_type = 'conv'

        dna.add_vertex(after_vertex_id, vertex_type, edge_type)
        return True
    
    def remove_convolution(self, dna):
        #随机选择一个dna删除所有边
        if len(dna.vertices) < 3:
            return False
        vertex_id = random.choice(range(1, len(dna.vertices) - 1))
        dna.remove_vertex(vertex_id)
        return True