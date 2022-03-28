# coding=utf-8

import os, sys
import redis
from compute.config import RedisConfig
from compute.log import Log
from compute.file import exec_cmd_remote
from compute.gpu import get_gpu_info

class PIDRedis():
    FORMAT_STR = 'EDL=>PID->|S:%s,SPID:%s,W:%s|'
    IDENTIFIER_STR = FORMAT_STR[:13]
    
    g = RedisConfig()
    REDIS_DB_IP = g.read_ini_file('log_server')
    REDIS_DB_PORT = g.read_ini_file('log_server_port')

    def __init__(self):
        pool = redis.ConnectionPool(host=self.REDIS_DB_IP, port=int(self.REDIS_DB_PORT), socket_connect_timeout=1)
        r = redis.Redis(connection_pool=pool, db=1)
        connection_flag = True
        try:
            r.ping()
        except Exception as e:
            connection_flag = False
            Log.warn('Connect redis error, please exit manually, errors:%s'%(str(e)))
            sys.exit()
        if connection_flag:
            pass 
        self.r = r
    
    def add_record(self, super_node_ip, super_node_pid, worker_node_ip, worker_node_pid):
        """Add a reord inot redis list
        Params
        ------
        - super_node_ip     (str): center server's ip
        - super_node_pid    (int): pid for running process in machine of `worker_node_ip`
        - worker_node_ip    (str): worker server's ip
        - worker_node_pid   (int): pid of running process in machine of `worker_node_ip`
        """
        _name = self.FORMAT_STR % (super_node_ip,super_node_pid,worker_node_ip)
        _pid = str(worker_node_pid).encode('utf-8')
        self.r.rpush(_name, _pid)

    def remove_record(self, super_node_ip, super_node_pid, worker_node_ip, worker_node_pid):
        """Add a reord inot redis list
        Params
        ------
        - super_node_ip     (str): center server's ip
        - super_node_pid    (int): pid for running process in machine of `worker_node_ip`
        - worker_node_ip    (str): worker server's ip
        - worker_node_pid   (int): pid of running process in machine of `worker_node_ip`
        """
        _name = self.FORMAT_STR % (super_node_ip,super_node_pid,worker_node_ip)
        _pid = str(worker_node_pid).encode('utf-8')
        self.r.lrem(_name, 0, _pid)
    
    def get_all_record(self, super_node_ip, super_node_pid, worker_node_ip, clear=True):
        """Get all records
        Params
        ------
        - super_node_ip     (str) : center server's ip
        - super_node_pid    (int) : pid for running process in machine of `worker_node_ip`
        - worker_node_ip    (str) : worker server's ip
        - clear             (bool): weather clear the buffer
        """
        # NOTE: risk for unlock.
        _name = self.FORMAT_STR % (super_node_ip,super_node_pid,worker_node_ip)
        _res = [int(_.decode('utf-8')) for _ in self.r.lrange(_name,0,-1)]
        if clear:
            self.r.delete(_name)
        return _res

    def get_all_keys(self):
        return [_.decode('utf-8') for _ in self.r.keys() if _.decode('utf-8').startswith(self.IDENTIFIER_STR)]
        
class PIDManager():
    _pid = os.getpid()
    gpu_info = get_gpu_info()

    class WorkerEnd:
        @staticmethod
        def add_worker_pid(super_node_ip, super_node_pid, worker_node_ip):
            """ Registry a process, only called by worker node
            """
            pidr = PIDRedis()
            pidr.add_record(super_node_ip, super_node_pid, worker_node_ip, PIDManager._pid)
        
        @staticmethod
        def remove_worker_pid(super_node_ip, super_node_pid, worker_node_ip):
            """Unregister a process
            """
            pidr = PIDRedis()
            pidr.remove_record(super_node_ip, super_node_pid, worker_node_ip, PIDManager._pid)

    class SuperEnd:
        @staticmethod
        def query_worker_tuple():
            pidr = PIDRedis()
            all_keys = pidr.get_all_keys()
            res = []
            for k in all_keys:
                S,SPID,W = k.split('|')[1].split(',')
                snip = S.split(':')[1]
                snpid = int(SPID.split(':')[1])
                wnip= W.split(':')[1]
                if snip == pidr.REDIS_DB_IP and snpid == PIDManager._pid:
                    queryed_pids = pidr.get_all_record(snip,snpid,wnip)
                    res += [ (snip,snpid,wnip,_) for _ in queryed_pids] 
            return res

        @staticmethod
        def kill_all_process():
            for _, _, worker_ip, worker_pid in PIDManager.SuperEnd.query_worker_tuple():
                ginfo = PIDManager.gpu_info[worker_ip]
                ssh_name = ginfo['ssh_name']
                ssh_passwd = ginfo['ssh_password']
                
                remote_cmd = 'sshpass -p {ssh_passwd} ssh {ssh_name}@{worker_ip} kill -TERM -- -{PID}'.format(
                    ssh_name = ssh_name,
                    ssh_passwd = ssh_passwd,
                    worker_ip = worker_ip,
                    PID = worker_pid
                )
                Log.debug('killing PID({PID}) in {worker_ip}'.format(PID=worker_pid, worker_ip = worker_ip))    
                _, std_err = exec_cmd_remote(remote_cmd, need_response=True)
            
                if std_err:
                    Log.warn(std_err)
                