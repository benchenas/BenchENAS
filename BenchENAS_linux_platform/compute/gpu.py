import os, time, multiprocess
import configparser
import subprocess
from compute.log import Log
import json
from compute.db import add_info, get_available_gpus, confirmed_used_gpu
import random


# 读compute/gpu.ini
def get_gpu_info():
    config_file_path = os.path.join(os.path.dirname(__file__), 'gpu.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    info = {}
    for sec in config.sections():
        try:
            worker_ip = config.get(sec, 'worker_ip')
            worker_name = config.get(sec, 'worker_name')
            ssh_name = config.get(sec, 'ssh_name')
            ssh_password = config.get(sec, 'ssh_password')
            gpu = [int(_id) for _id in config.get(sec, 'gpu').split(',')]
            content = {'worker_ip': worker_ip, 'worker_name': worker_name, 'ssh_name': ssh_name,
                       'ssh_password': ssh_password, 'gpu': gpu}
            info[sec] = content
        except BaseException as e:
            print(e)
    return info


def parse_nvidia_info(gpu_enabled_list, nvidia_info):
    # 2. split the output of the `nvidia-smi` command

    nvidia_info = nvidia_info.split('\n')
    list1 = list_temp = []
    list2 = []
    for line in nvidia_info:
        if line.startswith(' ' * 5):
            list_temp = list2
            continue
        list_temp += [line]

    # 3. confirm the gpu number according the showed gpus in the string
    num_gpu = 0
    remove_list = []
    for line in list1:
        line_split = line.split()
        if len(line_split) <= 1:
            remove_list.append(line)
        elif line_split[1] == str(num_gpu):
            num_gpu = num_gpu + 1
        elif line_split[1][-1] != "%":
            remove_list.append(line)
    for line in remove_list:
        list1.remove(line)

    # 4. confirm the job number
    num_gpu_list = []
    for i in range(0, num_gpu):
        num_gpu_list.append(str(i))

    job_num = 0
    remove_line = []
    for line in list2:
        line_split = line.split()
        if len(line_split) > 1 and line_split[1] in num_gpu_list:
            job_num = job_num + 1
        else:
            remove_line.append(line)
    for line in remove_line:
        list2.remove(line)

    job_num_list = [0] * num_gpu
    job_num = len(list2) - 7

    job_list = [[] for i in range(num_gpu)]

    if job_num != 0:
        # 4.1 peek each GPU job in this machine.
        for line in list2:
            process_info = line.split()
            job_in_gpu = int(process_info[1])  # GPU id of the job
            porcess_id = int(process_info[4])  # PID of the job
            process_type = process_info[5]  # Type of the job (`C` or `G`)
            process_usage_gpu_mem = int(process_info[-2][:-3])  # Mem usage of the job
            process_name = ' '.join(process_info[6:-2])  # job process name
            # store in job_list
            job_list[job_in_gpu] += [{
                'gpu_slot': job_in_gpu,
                'pid': porcess_id,
                'type': process_type,
                'process_name': process_name,
                'used_mem': process_usage_gpu_mem
            }]
            # counter increament
            job_num_list[job_in_gpu] += 1

    # 5. confirm the information of gpus in this machine
    gpus_info = []
    for i in range(num_gpu):
        if i not in gpu_enabled_list:
            continue

        gpu_model = list1[2 * i].split('|')[:]
        load_info = list1[2 * i + 1].split('|')[:]

        # 5.1 name of the GPU, such as 'GeForce GTX xxx'
        gpu_name = ' '.join(gpu_model[1].split()[1:-1])
        used_mem, total_mem = load_info[2].split('/')
        # 5.2 used mem in this gpu
        used_mem = used_mem.split()[0]
        used_mem = used_mem[:len(used_mem) - 3]
        # 5.3 total number of the gpu
        total_mem = total_mem.split()[0]
        total_mem = total_mem[:len(total_mem) - 3]
        total_mem = int(total_mem)
        used_mem = int(used_mem)
        # 5.4 rest gpu memory.
        left_mem = total_mem - used_mem

        # 5.5 extra for statistic
        Fan, Temp, _, CurPower, _, MaxPower = load_info[1].split()
        CoreLoad = load_info[3].split()[0]

        # add the result list.
        gpus_info += [{
            'name': gpu_name,
            'gpu_slot': i,
            'total_mem': total_mem,
            'used_mem': used_mem,
            'left_mem': left_mem,
            'job_num': job_num_list[i],  # this
            'fan': Fan,
            'temp': Temp,
            'cur_power': CurPower,
            'max_power': MaxPower,
            'core_load': CoreLoad,
            'job_list': json.dumps(job_list[i])  # job list, i.e., the Processes List in `nvidia-smi`.
        }]
    return gpus_info


# def parse_nvidia_info(gpu_enabled_list, nvidia_info):
#
#     # 2. split the output of the `nvidia-smi` command
#     nvidia_info=nvidia_info.split('\n')
#     list1=list_temp=[]
#     list2=[]
#     for line in nvidia_info:
#         if line.startswith(' '*5):
#             list_temp=list2
#             continue
#         list_temp+=[line]
#     # 3. confirm the gpu number according the showed gpus in the string
#     num_gpu=(len(list1)-7)//4
#
#     # 4. confirm the job number
#     job_num_list=[0]*num_gpu
#     job_num=len(list2)-7
#     job_list=[ [] for i in range(num_gpu)]
#     if job_num==1 and list2[5].split()[1]=='No':
#         job_num=0
#         # print('no job found')
#     else:
#         # 4.1 peek each GPU job in this machine.
#         for i in range(job_num):
#             process_info=list2[5+i].split()
#             print(process_info)
#             job_in_gpu=int(process_info[1]) # GPU id of the job
#             porcess_id=int(process_info[4]) # PID of the job
#             process_type=process_info[5] # Type of the job (`C` or `G`)
#             process_usage_gpu_mem=int(process_info[-2][:-3]) # Mem usage of the job
#             process_name=' '.join(process_info[6:-2]) # job process name
#             # store in job_list
#             job_list[job_in_gpu]+=[{
#                 'gpu_slot':job_in_gpu,
#                 'pid':porcess_id,
#                 'type':process_type,
#                 'process_name':process_name,
#                 'used_mem':process_usage_gpu_mem
#             }]
#             # counter increament
#             job_num_list[job_in_gpu]+=1
#
#
#     # 5. confirm the information of gpus in this machine
#     gpus_info=[]
#     for i in range(num_gpu):
#         if i not in gpu_enabled_list:
#             continue
#
#         gpu_model=list1[8+4*i].split('|')[:]
#         load_info=list1[9+4*i].split('|')[:]
#
#         # 5.1 name of the GPU, such as 'GeForce GTX xxx'
#         gpu_name=' '.join(gpu_model[1].split()[1:-1])
#         used_mem,total_mem=load_info[2].split('/')
#         # 5.2 used mem in this gpu
#         used_mem=used_mem.split()[0]
#         used_mem=used_mem[:len(used_mem)-3]
#         # 5.3 total number of the gpu
#         total_mem=total_mem.split()[0]
#         total_mem=total_mem[:len(total_mem)-3]
#         total_mem=int(total_mem)
#         used_mem=int(used_mem)
#         # 5.4 rest gpu memory.
#         left_mem=total_mem-used_mem
#
#         # 5.5 extra for statistic
#         Fan,Temp,_,CurPower,_,MaxPower=load_info[1].split()
#         CoreLoad=load_info[3].split()[0]
#         # add the result list.
#         gpus_info+=[{
#             'name':gpu_name,
#             'gpu_slot':i,
#             'total_mem':total_mem,
#             'used_mem':used_mem,
#             'left_mem':left_mem,
#             'job_num':job_num_list[i], # this
#             'fan':Fan,
#             'temp':Temp,
#             'cur_power': CurPower,
#             'max_power': MaxPower,
#             'core_load':CoreLoad,
#             'job_list': json.dumps(job_list[i]) # job list, i.e., the Processes List in `nvidia-smi`.
#             }]
#     return gpus_info


def detect_gpu():
    gpu_info = get_gpu_info()
    info = []
    available_num = 0
    for each_work in gpu_info.keys():
        ip = gpu_info[each_work]['worker_ip']
        Log.debug('Begin to detect on %s' % (ip))
        worker_name = gpu_info[each_work]['worker_name']
        ssh_name = gpu_info[each_work]['ssh_name']
        ssh_password = gpu_info[each_work]['ssh_password']
        gpus = gpu_info[each_work]['gpu']

        _cmd = 'sshpass -p \'%s\' ssh %s@%s nvidia-smi' % (ssh_password, ssh_name, worker_name)
        nvidia_info = subprocess.Popen(_cmd, stdout=subprocess.PIPE, shell=True).stdout.read().decode()
        Log.debug('Perform the cmd \'%s\'' % (_cmd))
        Log.debug('Response:\n%s' % (nvidia_info))
        Log.debug('Start to parse the gpu info from %s for operating the tables' % (ip))
        gpus_info = parse_nvidia_info(gpus, nvidia_info)

        for each_info in gpus_info:
            _item = {
                'worker_ip': ip,
                'gpu_id': each_info['gpu_slot'],
                'status': 0 if each_info['job_num'] == 0 else 1,
                'remark': each_info['job_list']
            }
            info.append(_item)
            if _item['status'] == 0:
                available_num = available_num + 1
    Log.info('%d available GPUs have been detected' % (available_num))
    Log.debug('Start to operate the database tables')
    add_info(gpu_info, info)


def locate_gpu_to_be_used():
    available_gpus = get_available_gpus()
    # currently, we randomly select one 
    if len(available_gpus) == 0:
        Log.info('No available GPUs currently')
        return None
    else:
        # Log.info('%d available GPUs currently'%(len(available_gpus)))
        random.shuffle(available_gpus)
        located_gpu = available_gpus[0]
        # delete this located gpu from the gpu_list
        Log.debug('Delete the used gpu from the list')
        confirmed_used_gpu([str(located_gpu[0])])
        chosen_gpu = {'worker_ip': located_gpu[1], 'gpu_id': located_gpu[2], 'ssh_name': located_gpu[3],
                      'ssh_password': located_gpu[4]}
        Log.info('%d available GPUs, the selected is: GPU#%d on %s' % (
        len(available_gpus), int(chosen_gpu['gpu_id']), chosen_gpu['worker_ip']))
        return chosen_gpu


def gpus_all_available():
    assigned_gpus = get_gpu_info()
    num_assigned_gpus = 0
    for each_item in assigned_gpus.values():
        num_assigned_gpus += len(each_item['gpu'])
    Log.debug('%d GPUs were assigned for this experiment' % (num_assigned_gpus))

    num_available_gpus_in_db = len(get_available_gpus())

    Log.debug('%d GPUS were available in the database' % (num_available_gpus_in_db))

    if num_assigned_gpus == num_available_gpus_in_db:
        return True
    else:
        return False


def run_detect_gpu():
    Log.info('Start to detect GPUs')

    def fun1():
        while True:
            Log.debug('start to periodicity detect GPU ...')
            detect_gpu()
            time.sleep(60)

    p = multiprocess.Process(target=fun1, args=())  # start to detect GPUs
    p.start()


if __name__ == '__main__':
    print(get_gpu_info())
