<p align="center">
 <img width="100%" src="https://github.com/benchenas/BenchENAS/blob/master/attach/logo.PNG"/>
</p>

<p align="center">
   <a href="https://github.com/benchenas/BenchENAS/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/benchENAS/1.1.1/">
    <img src="https://img.shields.io/pypi/v/benchENAS?color=blue&label=pypi" alt="CircleCI" />
  </a>
    <a href="https://benchenas.com/api/index.html">
    <img src="https://img.shields.io/website/http/benchenas.com/api/index.html.svg?down_color=red&down_message=offline&up_message=online" />
  </a>
    <a href="https://ieeexplore.ieee.org/document/9697075">
    <img src="https://img.shields.io/badge/ArXiv-2108.03856-orange.svg"/>
  </a>
 </p>

**[Documentation](https://benchenas.com/api/index.html)** | **[Website](https://benchenas.com/)** 

This is the python package of BenchENAS.
Here's a quick run down of the main steps of installing and running BenchENAS. For more details see our [Documentation](https://benchenas.com/api/modules.html/) and [Example](https://benchenas.com/).


## Installation

```
$ pip install benchENAS
```

## Tesing

```
$ pytest
```

## Running

1. Start the redis-server on the center computer

```
$ ./redis-server   redis.conf
```

2. Initialize configuration of the algorithm and training parameters

```python
alg_list = {'algorithm': 'aecnn', 'max_gen': 20, 'pop_size': 20,
            'log_server': 'xx.xx.xx.xx', 'log_server_port': 6379,
            'exe_path': '/home/xxx/anaconda3/bin/python3'}
            
# use dataset in [MNIST, CIFAR10, CIFAR100]
train_list = {'dataset': 'CIFAR10', 'optimizer': 'SGD', 'lr': 0.025,
              'batch_size': 64, 'total_epoch': 50, 'lr_strategy': 'ExponentialLR'}

# use customized dataset, the directory of dataset refers to eye_dataset[BenchENAS/example/eye_dataset]
train_list = {'dataset': 'customized', 'data_dir': '/home/xxx/xx_dataset', 
              'img_input_size': [244, 244, 3], 'optimizer': 'SGD', 'lr': 0.025,
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
<!-- 
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

## Contributing
This is an open-source project welcoming your contributions. You can contribute in three ways:
- Star and fork BenchENAS to follow its latest developments, share it with your networks, and [ask questions](https://github.com/benchenas/BenchENAS/issues) about it.
- Use BenchENAS in your project and let us know any bugs (& fixes) and feature requests/suggestions via [creating an issue](https://github.com/benchenas/BenchENAS/issues).
- Contribute via branch, fork, and pull for minor fixes and new features, functions, or algorithms to become one of the [contributors](https://github.com/benchenas/BenchENAS/blob/master/CONTRIBUTING.md).

## Citing
View the [published paper](https://ieeexplore.ieee.org/document/9697075). If you use or reference BenchENAS, please cite:
```python
@ARTICLE{9697075,
  author={Xie, Xiangning and Liu, Yuqiao and Sun, Yanan and Yen, Gary G. and Xue, Bing and Zhang, Mengjie},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={BenchENAS: A Benchmarking Platform for Evolutionary Neural Architecture Search}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TEVC.2022.3147526}}
```

## License

- [MIT License](https://github.com/benchenas/BenchENAS/blob/master/LICENSE) -->

