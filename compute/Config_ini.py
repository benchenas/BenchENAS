import os

algorithm = ''
dataset = ''
alg_name = ''
max_gen = 0
pop_size = 0
log_server = ''
log_server_port = 0
exe_path = ''
print(exe_path)
optimizer = ''
batch_size = 0
total_epoch = 0
lr = 0.0
lr_strategy = ''

gpu_info = {}
dest_dir = ''


def amend(alg_list, train_list, gpu_info_list):
    global algorithm
    global alg_name
    global max_gen
    global pop_size
    global log_server
    global log_server_port
    global exe_path
    global dataset
    global optimizer
    global batch_size
    global total_epoch
    global lr
    global lr_strategy
    global gpu_info
    global dest_dir
    algorithm = alg_list['algorithm']
    dataset = train_list['dataset']
    alg_name = '_'.join([algorithm, dataset])
    max_gen = alg_list['max_gen']
    pop_size = alg_list['pop_size']
    log_server = alg_list['log_server']
    log_server_port = alg_list['log_server_port']
    exe_path = alg_list['exe_path']
    print(exe_path)
    optimizer = train_list['optimizer']
    batch_size = train_list['batch_size']
    total_epoch = train_list['total_epoch']
    lr = train_list['lr']
    lr_strategy = train_list['lr_strategy']
    gpu_info = gpu_info_list

