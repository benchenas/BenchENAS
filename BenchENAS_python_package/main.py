from compute import Config_ini
from compute.Config_ini import amend

from compute.log import Log
import time
import importlib
from read_best_acc import read_result, write_result


def run(alg_list, train_list, gpu_info_list, search_space='micro'):
    amend(alg_list, train_list, gpu_info_list)
    from compute.pid_manager import PIDManager
    alg_name = alg_list['algorithm']
    name = '_'.join([alg_name, train_list['dataset']])
    algs = importlib.import_module('algs.' + alg_name + '.main')
    try:
        if alg_name != 'nsga_net':
            r = algs.Run(alg_list, train_list, gpu_info_list)
        else:
            r = algs.Run(alg_list, train_list, gpu_info_list, search_space)
        r.do()
        best_acc = read_result(name)
        print(best_acc)
        if alg_name == 'nsga_net':
            alg_name = alg_name + '_' + search_space

        write_result(alg_name, best_acc)
    except KeyboardInterrupt:
        Log.info('Receive KeyboardInterrupt SIGNAL')
        Log.info('Wait to exit')
        time.sleep(1)
        Log.info('Kill processes in workers')
        PIDManager.SuperEnd.kill_all_process()
        Log.info('Good bye')


# if __name__ == '__main__':
#     alg_list = {'algorithm': 'hierarchical_representations', 'max_gen': 20, 'pop_size': 20,
#                 'log_server': 'xx.xx.xx.xx', 'log_server_port': 6379,
#                 'exe_path': '/home/xxx/anaconda3/bin/python3'}
#
#     train_list = {'dataset': 'customized', 'data_dir': '/home/xiaoyang/eye_dataset',
#                  'img_input_size': [244, 244, 3], 'optimizer': 'SGD', 'lr': 0.025,
#                  'batch_size': 64, 'total_epoch': 5, 'lr_strategy': 'ExponentialLR'}
#                               
#     gpu_info_list = {}
#     content = {'worker_ip': 'xx.xx.xx.xx', 'worker_name': 'cuda', 'ssh_name': 'xxx',
#                'ssh_password': '.123456', 'gpu': [1, 2]}
#     gpu_info_list['xx.xx.xx.xx'] = content
#     run(alg_list, train_list, gpu_info_list)
