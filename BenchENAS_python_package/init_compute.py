'''
this file is to initialize the compute platform, make sure to run this file before run the algorithm
'''
from compute import Config_ini
from compute.Config_ini import amend
from compute.db import init_db
from compute.gpu import run_detect_gpu

from compute.file import init_work_dir_on_all_workers


def start_compute_platform():
    init_db()
    from compute.redis import RedisLog
    RedisLog.run_dispatch_service()
    init_work_dir_on_all_workers()
    run_detect_gpu()


def run(alg_list, train_list, gpu_info_list):
    amend(alg_list, train_list, gpu_info_list)
    start_compute_platform()


# if __name__ == '__main__':
#     alg_list = {'algorithm': 'aecnn', 'max_gen': 20, 'pop_size': 20,
#                 'log_server': '192.168.1.1', 'log_server_port': 6379}
# 
#     train_list = {'dataset': 'CIFAR10', 'data_dir': 'E:\\PYPI\\eye_dataset',
#                   'img_input_size': [244, 244, 3], 'optimizer': 'SGD', 'lr': 0.025,
#                   'batch_size': 64, 'total_epoch': 50, 'lr_strategy': 'ExponentialLR'}
# 
#     gpu_info_list = {}
#     content1 = {'worker_ip': '192.168.1.1', 'worker_name': 'cuda', 'ssh_name': 'xxxx',
#                 'ssh_password': '123456', 'gpu': [0], 'port': 22,
#                 'exe_path': '/home/xiaoyang/anaconda3/bin/python3'}
#      gpu_info_list['192.168.1.1'] = content1
#     run(alg_list, train_list, gpu_info_list)

