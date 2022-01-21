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

- The codes have been tested on Python 3.6 + pytorch 1.1 + torchvision 0.3.0 (pytorch 1.3 seems also ok, but not test thoroughly)



- # Requirements:

 - Center Computer:
   - redis (ubuntu software, start using the command redis-server --protected-mode on)
   - sshpass (python lib)

 - Conter Computer & workers:
   - multiprocess (python lib)
   - redis (python lib)
 
 
- # How to use
- Start the redis-server on the center computer (redis-server --protected-mode no)
- Start the init_compute.py script to start the compute platform
- Start the algorithm you would like to perform

--------------------------------------------------------------------------------

**Citing**


If you find *BenchENAS* useful in your research, please consider adding the following citation:

```bibtex
@inproceedings{benchenas,
               author = {Benedek Rozemberczki and Paul Scherer and Yixuan He and George Panagopoulos and Alexander Riedel and Maria Astefanoaei and Oliver Kiss and Ferenc Beres and Guzman Lopez and Nicolas Collignon and Rik Sarkar},
               title = {{PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models}},
               year = {2021},
               booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
               pages = {4564–4573},
}
```

--------------------------------------------------------------------------------

