'''
this file is to initialize the compute platform, make sure to run this file before run the algorithm
'''
from compute.db import init_db
from compute.gpu import run_detect_gpu
from compute.redis import RedisLog
from compute.file import init_work_dir_on_all_workers


def start_compute_platform():
    init_db()  # initialize the database
    RedisLog.run_dispatch_service()
    init_work_dir_on_all_workers()  # initialize the working directory on each worker
    run_detect_gpu()


if __name__ == '__main__':
    start_compute_platform()
