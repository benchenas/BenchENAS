<p align="center">
 <img width="100%" src="attach/logo.PNG"/>
</p>

<p align="center">
   <a href="https://github.com/benchenas/BenchENA/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/benchENAS/1.1.1/">
    <img src="https://img.shields.io/pypi/v/benchENAS?color=blue&label=pypi" alt="CircleCI" />
  </a>
    <a href="https://github.com/benchenas/BenchENA/actions">
    <img src="https://github.com/benchenas/BenchENA/workflows/CI/badge.svg" alt="Coverage" />
  </a>
    <a href="https://benchenas.com/api/index.html">
    <img src="https://img.shields.io/website/http/benchenas.com/api/index.html.svg?down_color=red&down_message=offline&up_message=online" />
  </a>
    <a href="https://arxiv.org/abs/2108.03856">
    <img src="https://img.shields.io/badge/ArXiv-2108.03856-orange.svg"/>
  </a>
 </p>

**[Documentation](https://benchenas.com/api/index.html)** | **[Website](https://benchenas.com/)** 

*BenchENAS* is a benchmarking platform to conduct fair comparisons upon Evolutionary algorithm based Neural Architecture Search (ENAS) algorithms.

<p align="justify">Nine representative state-of-the-art ENAS algorithms, popular data processing techniques for 3 widely used benchmark datasets, as well as configurable trainer settings such as learning rate policy, optimizers, batch size, and training epochs, have been implemented intothe proposed BenchENAS platform. To this end, the related researchers can illustrate the innovativeness of their proposed algorithms by making fair comparisons with the state-of-the-art ENAS algorithms. Furthermore, An efficient parallel component and a cache component are designed to accelerate the fitness evaluation phase in BenchENAS.</p>

The parallel component is based on the parallel training mechanism of existing deep learning libraries and can be used to speed up the running of the corresponding ENAS algorithm. 

 The cache component is used to record the fitness values for each architecture and to reuse the fitness values in the cache when an individual of the same architecture
appears.


## Installation

#### From pip

Install using ``` pip install benchENAS ```

#### From Source

Download this repository into your project folder.

## Tesing

```
$ pytest
```

## Running

Here's a quick run down of the main steps of running BenchENAS. For more details see our [Documentation](https://benchenas.com/api/modules.html/) and [Example](https://benchenas.com/).

1. Start the redis-server on the center computer

```
$ ./redis-server   redis.conf
```

2. Initialize configuration of the algorithm and training parameters

```python
alg_list = {'algorithm': 'aecnn', 'max_gen': 20, 'pop_size': 20,
            'log_server': 'xx.xx.xx.xx', 'log_server_port': 6379,
            'exe_path': '/home/xxx/anaconda3/bin/python3'}
train_list = {'dataset': 'CIFAR10', 'optimizer': 'SGD', 'lr': 0.025,
              'batch_size': 64, 'total_epoch': 50, 'lr_strategy': 'ExponentialLR'}

gpu_info_list = {}
content = {'worker_ip': 'xx.xx.xx.xx', 'worker_name': 'cuda0', 'ssh_name': 'xxx',
           'ssh_password': '.123456', 'gpu': [1, 2]}
gpu_info_list['xx.xx.xx.xx'] = content
```

3. Start the init_compute.py script to start the compute platform and detect free GPU devices

```python
from benchenas import init_compute

init_compute.run(alg_list, train_list, gpu_info_list)
```

4. Start the algorithm you would like to perform

```python
from benchenas import main

main.run(alg_list, train_list, gpu_info_list)
```

## Algorithms included

- AE_CNN: [Completely automated CNN architecture design based on blocks](https://ieeexplore.ieee.org/document/8742788)

- CGP_CNN: [A Genetic Programming Approach to Designing Convolutional Neural Network Architecture](https://arxiv.org/abs/1704.00764)

- CNN_GA: [Automatically Designing CNN Architectures Using the Genetic Algorithm for Image Classification](https://arxiv.org/abs/1808.03818)

- Evo_CNN: [Evolving deep convolutional neural networks for image classification](https://arxiv.org/abs/1803.06492)

- Genetic_CNN: [Genetic CNN](https://arxiv.org/abs/1703.01513v1)

- Hierarchical_Representations: [Hierarchical Representations for Efficient Architecture Search](https://arxiv.org/abs/1711.00436)

- Large_Scale: [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041)

- NSGA-Net: [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://arxiv.org/abs/1810.03522v2)

- Regularized_Evo: [Regularized Evolution for Image Classfier Architecture Search](https://arxiv.org/abs/1802.01548v7)

## License

- [MIT License](https://github.com/benchenas/BenchENA/blob/master/LICENSE)

