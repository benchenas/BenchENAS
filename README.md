<p align="center">
 <img width="100%" src="attach/logo.PNG"/>
</p>

<p align="center">
   <a href="https://github.com/benchenas/BenchENAS/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="CircleCI" />
  </a>
   <a href="https://pypi.org/project/benchENAS/1.1.1/">
    <img src="https://img.shields.io/pypi/v/benchENAS?color=blue&label=pypi" alt="CircleCI" />
  </a>
    <a href="https://github.com/benchenas/BenchENAS/actions">
    <img src="https://github.com/benchenas/BenchENA/workflows/CI/badge.svg" alt="Coverage" />
  </a>
    <a href="https://benchenas.com/api/index.html">
    <img src="https://img.shields.io/website/http/benchenas.com/api/index.html.svg?down_color=red&down_message=offline&up_message=online" />
  </a>
    <a href="https://ieeexplore.ieee.org/document/9697075">
    <img src="https://img.shields.io/badge/ArXiv-2108.03856-orange.svg"/>
  </a>
 </p>

**[Documentation](https://benchenas.com/api/index.html)** | **[Website](https://benchenas.com/)** 

*BenchENAS* is a benchmarking platform to conduct fair comparisons upon Evolutionary algorithm based Neural Architecture Search (ENAS) algorithms.

<p align="justify">Nine representative state-of-the-art ENAS algorithms, popular data processing techniques for 3 widely used benchmark datasets and customized dataset, as well as configurable trainer settings such as learning rate policy, optimizers, batch size, and training epochs, have been implemented intothe proposed BenchENAS platform. To this end, the related researchers can illustrate the innovativeness of their proposed algorithms by making fair comparisons with the state-of-the-art ENAS algorithms.  </p>

Furthermore, An efficient parallel component and a cache component are designed to accelerate the fitness evaluation phase in BenchENAS. The parallel component is based on the parallel training mechanism of existing deep learning libraries and can be used to speed up the running of the corresponding ENAS algorithm. The cache component is used to record the fitness values for each architecture and to reuse the fitness values in the cache when an individual of the same architecture
appears.


## Installation

#### From python package

Please click [Here](https://github.com/benchenas/BenchENAS/BenchENAS_python_package).

#### From linux source code 

Please click [Here](https://github.com/benchenas/BenchENAS/BenchENAS_linux_source_code).

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

- [MIT License](https://github.com/benchenas/BenchENAS/blob/master/LICENSE)

