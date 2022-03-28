import redis
from compute.config import RedisConfig
import os,time,multiprocess,logging,sys
import json
from compute.log import Log
from compute.file import get_algo_local_dir,get_population_dir

def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')

    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
 
    l.setLevel(level)
    l.addHandler(fileHandler)
 
    return logging.getLogger(logger_name)

class RedisLog(object):

    MSG_TYPE = ['LOG','WRITE_FILE']
    
    def __init__(self,name):
        g = RedisConfig()
        db_ip = g.read_ini_file('log_server')
        db_port = g.read_ini_file('log_server_port')
        pool = redis.ConnectionPool(host=db_ip, port=int(db_port), socket_connect_timeout=1)
        r = redis.Redis(connection_pool=pool,db=1)
        
        connection_flag = True
        try:
            r.ping()
        except Exception as e:
            connection_flag = False
            Log.warn('Connect redis error, please exit manually, errors:%s'%(str(e)))
            sys.exit()
        if connection_flag:
            Log.info('Connect redis successfully...')
        self.r = r
        self.name=name

    def info(self, info):
        self._writedb('LOG', info)
    
    def write_file(self, fdir, fname, data):
        content={'fdir':fdir,'fname':fname,'data':data}
        self._writedb('WRITE_FILE',content)
    
    def _writedb(self,msg_type,content):
        assert msg_type in self.MSG_TYPE
        v= {'name':self.name,'type':msg_type,'content':content}
        v= json.dumps(v).encode('utf-8')
        self.r.rpush('RUN', v)
        
    def _readdb(self):
        info = self.r.lpop('RUN')
        if info is None:
            return None
        # print(info)
        info = json.loads(info.decode('utf-8'))
        return info
    
    @staticmethod
    def run_dispatch_service():
        Log.info('Start to read message from Redis')
        def proc_func():
            rdb=RedisLog('')
            log_dict={}
            log_dir= os.path.join(get_algo_local_dir(), 'log')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            while True:
                data=rdb._readdb()
                if data is not None:
                    name, dtype, content = data['name'], data['type'], data['content']
                    Log.debug('Redis is reading: name:%s, type:%s, content:%s'%(name, dtype, content))
                    if dtype == 'LOG': 
                        # create logger.
                        if name not in log_dict:
                            log_file=os.path.join(log_dir, name)
                            logger=get_logger(name,log_file)
                            log_dict[name]=logger

                        logger = log_dict[name]
                        logger.info(content)
                        # print(content)

                    elif dtype == 'WRITE_FILE':
                        fdir,fname,fdata = content['fdir'],content['fname'],content['data']
                        
                        if fdir == 'CACHE' or fdir == 'RESULTS':
                            fdir = get_population_dir()
                        
                        if not os.path.exists(fdir):
                            os.makedirs(fdir)
                        with open(os.path.join(fdir,fname),'a+') as f:
                            f.write(fdata)
                            f.flush()
                    else:
                        assert 0

                time.sleep(1)

        proc=multiprocess.Process(target=proc_func)
        proc.start()
        




