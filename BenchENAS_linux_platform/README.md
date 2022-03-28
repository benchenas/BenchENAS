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

This is the linux source code of BenchENAS.
Here's a quick run down of the main steps of installing and running BenchENAS. For more details see our [Documentation](https://benchenas.com/api/modules.html/) and [Example](https://benchenas.com/).

## Installation

1. Download the source code.
```
$ git clone https://github.com/benchenas/BenchENAS.git
$ cd BenchENAS_linux_platform
```
2. Install sshpass.
```
$ sudo apt-get install sshpass
```
3. Install redis and start redis server.
```
$ sudo apt-get install redis-server
$ redis-server --protected-mode on
```
4. Installation of python third-party libraries.
```
$ pip install redis
$ pip install paramiko
```

## Running

1. Start the init_compute.py script to start the platform.

```
$ python main.py
```

2. Start the algorithm you would like to perform.

```
$ python init_compute.py
```
