import time
from compute.log import Log
from compute.gpu import locate_gpu_to_be_used, get_gpu_info
from compute.file import transfer_file_relative, exec_python, get_algo_local_dir
from compute.pid_manager import PIDRedis, PIDManager


def dispatch_to_do(_id, _uuid):
    gpu_info = locate_gpu_to_be_used()

    while gpu_info is None:
        time.sleep(60)
        gpu_info = locate_gpu_to_be_used()

    if gpu_info is not None:
        worker_ip, gpu_id, ssh_name, ssh_password = gpu_info['worker_ip'], \
                                                    gpu_info['gpu_id'], \
                                                    gpu_info['ssh_name'], \
                                                    gpu_info['ssh_password']
        worker = get_gpu_info()
        worker_name = worker[worker_ip]['worker_name']
        # 1. transfer file
        # NOTE: commom files shall be uploaded when initialization instead of here
        indi_file_name = '%s.py' % (_id)
        sync_file_list = [
            ('%s/scripts/%s' % (get_algo_local_dir(), indi_file_name), indi_file_name)
        ]
        for src, dst in sync_file_list:
            transfer_file_relative(ssh_name, ssh_password, worker_ip, worker_name, src, dst)
        Log.info('Transfer file successfully...')
        # 2. execuate the file
        exec_python(ssh_name, ssh_password, worker_ip,  worker_name, 'training.py',
                    args={
                        '-gpu_id': str(gpu_id),
                        '-file_id': _id,
                        '-uuid': _uuid,
                        '-super_node_ip': PIDRedis.REDIS_DB_IP,
                        '-super_node_pid': str(PIDManager._pid),
                        '-worker_node_ip': worker_ip
                    },
                    )
