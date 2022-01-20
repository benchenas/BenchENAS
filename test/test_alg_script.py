import importlib

import numpy as np
import torch


from comm.log import Log
from compute import Config_ini


def config_param(alg):
    Config_ini.dataset = 'CIFAR10'
    Config_ini.pop_size = 1
    Config_ini.max_gen = 2
    Config_ini.alg_name = alg + '_' + Config_ini.dataset


def test_alg_script():
    """
        Testing pytorch scripts in setting environment
    """
    algs = ['aecnn', 'cnn_ga', 'genetic_CNN', 'hierarchical_representations',
            'large_scale']
    batch_size = 16
    shuffle = True

    for alg in algs:
        config_param(alg)

        s = importlib.import_module('algs.' + alg + '.genetic.' + 'statusupdatetool')
        p = importlib.import_module('algs.' + alg + '.genetic.' + 'population')
        g = importlib.import_module('algs.' + alg + '.genetic.' + 'evaluate')

        params = s.StatusUpdateTool.get_init_params()
        pops = p.Population(params=params, gen_no=0)
        pops.initialize()
        fitness = g.FitnessEvaluate(pops.individuals, Log)
        fitness.generate_to_python_file(test=True)
        script = importlib.import_module('example.' + alg + '_indi00000_00000')

        inputs = torch.rand([batch_size, 3, 32, 32])
        targets = np.floor(10 * torch.rand(16))
        net = script.EvoCNNModel()
        train_dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        for _, data in enumerate(train_loader, 0):
            x, _ = data
            out = net(x)
        assert out.shape[0] == x.shape[0]
        assert out.shape[1] == 10


def test_cgp_script():
    config_param('cgp_cnn')
    batch_size = 16
    shuffle = True

    from algs.cgp_cnn.cgp_config import CgpInfoConvSet
    from algs.cgp_cnn.genetic.statusupdatetool import StatusUpdateTool
    from algs.cgp_cnn.genetic.population import Population
    from algs.cgp_cnn.genetic.evaluate import FitnessEvaluate

    params = StatusUpdateTool.get_init_params()
    params['lam'] = 0
    network_info = CgpInfoConvSet(params)
    init = False
    pops = Population(0, params, network_info, init)
    pops.initialize()
    fitness = FitnessEvaluate(pops.individuals, Log)
    fitness.generate_to_python_file(test=True)
    script = importlib.import_module('example.' + 'cgp_cnn_indi00000_parent')

    inputs = torch.rand([batch_size, 3, 32, 32])
    targets = np.floor(10 * torch.rand(16))
    net = script.EvoCNNModel()
    train_dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    for _, data in enumerate(train_loader, 0):
        x, _ = data
        out = net(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == 10


def test_nsga_script():
    config_param('nsga_net')
    search_spaces = ['micro', 'macro']
    batch_size = 16
    shuffle = True

    from algs.nsga_net.genetic.population import Population
    from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
    from algs.nsga_net.genetic.evaluate import FitnessEvaluate
    from algs.nsga_net.utils.flops_counter import calculate_flop

    for search_space in search_spaces:
        StatusUpdateTool.change_search_space(search_space)
        params = StatusUpdateTool.get_init_params()
        pops = Population(gen_no=0, params=params)
        pops.initialize()
        fitness = FitnessEvaluate(pops.individuals, params, Log)
        fitness.generate_to_python_file(test=True)
        script = importlib.import_module('example.' + 'nsga_' + search_space + '_indi00000_00000')

        inputs = torch.rand([batch_size, 3, 32, 32])
        targets = np.floor(10 * torch.rand(16))
        net = script.EvoCNNModel()
        train_dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        for _, data in enumerate(train_loader, 0):
            x, _ = data
            out = net(x)
        assert out.shape[0] == x.shape[0]
        assert out.shape[1] == 10

    for search_space in search_spaces:
        StatusUpdateTool.change_search_space(search_space)
        for indi in pops.individuals:
            indi.flop = calculate_flop(indi)
            assert isinstance(indi.flop, float)


def test_regularized_evo_script():
    config_param('regularized_evolution')
    batch_size = 16
    shuffle = True

    from algs.regularized_evolution.genetic.statusupdatetool import StatusUpdateTool
    from algs.regularized_evolution.genetic.population import Population
    from algs.regularized_evolution.genetic.evaluate import FitnessEvaluate

    params = StatusUpdateTool.get_init_params()
    pops = Population(gen_no=0, params=params)
    pops.initialize()
    fitness = FitnessEvaluate(pops.individuals, params, Log)
    fitness.generate_to_python_file(test=True)
    script = importlib.import_module('example.' + 'regularized_evolution_indi00000_00000')

    inputs = torch.rand([batch_size, 3, 32, 32])
    targets = np.floor(10 * torch.rand(16))
    net = script.EvoCNNModel()
    train_dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    for _, data in enumerate(train_loader, 0):
        x, _ = data
        out = net(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == 10