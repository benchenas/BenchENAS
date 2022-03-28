import os
import multiprocessing
from compute.log import Log
from compute.file import get_population_dir


class CacheUtils(object):
    _lock = multiprocessing.Lock()
    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock
    
    @classmethod
    def load_cache_data(cls):
        file_name = os.path.join(get_population_dir(),'cache.txt')
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f'%(float(rs_[1]))
            f.close()
        return _map
    
    @classmethod
    def save_fitness_to_cache(cls, uuid, _acc):
        Log.info('Add record into cache, id:%s, acc:%.5f'%(uuid, _acc))
        f = open(os.path.join(get_population_dir(),'cache.txt'), 'a+')
        _str = '%s;%.5f\n'%(uuid, _acc)
        f.write(_str)
        f.close()


        
if __name__ == '__main__':
    print(os.path.dirname(os.path.dirname(__file__)))