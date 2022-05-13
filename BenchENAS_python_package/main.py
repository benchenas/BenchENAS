from compute import Config_ini
from compute.Config_ini import amend

from compute.log import Log
import time
import importlib
from read_best_acc import read_result, write_result


def run(alg_list, train_list, gpu_info_list, search_space='micro'):
    amend(alg_list, train_list, gpu_info_list)
    from compute.pid_manager import PIDManager
    alg = alg_list['algorithm']
    name = Config_ini.alg_name
    algs = importlib.import_module('algs.' + alg + '.main')
    try:
        if alg != 'nsga_net':
            r = algs.Run(alg_list, train_list, gpu_info_list)
        else:
            r = algs.Run(alg_list, train_list, gpu_info_list, search_space)
        r.do()
        best_acc = read_result(name)
        print('best_acc: ', best_acc)
        if alg == 'nsga_net':
            alg = alg + '_' + search_space

        write_result(alg, train_list['dataset'], best_acc)
    except KeyboardInterrupt:
        Log.info('Receive KeyboardInterrupt SIGNAL')
        Log.info('Wait to exit')
        time.sleep(1)
        Log.info('Kill processes in workers')
        PIDManager.SuperEnd.kill_all_process()
        Log.info('Good bye')


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
