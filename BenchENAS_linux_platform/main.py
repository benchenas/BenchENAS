from algs.nsga_net.utils.statusupdatetool import StatusUpdateTool
from compute.pid_manager import PIDManager
from compute.log import Log
import time
import importlib
from comm.utils import PlatENASConfig
from read_best_acc import read_result, write_result


if __name__ == '__main__':
    g = PlatENASConfig('algorithm')
    algs_name = g.read_ini_file('run_algorithm')
    name = g.read_ini_file('name')
    algs = importlib.import_module('algs.' + algs_name + '.main')
    try:
        r = algs.Run()
        r.do()
        best_acc = read_result(name)
        if algs_name == 'nsga_net':
            search_space = StatusUpdateTool.get_search_space()
            alg_name = algs_name + '_' + search_space

        write_result(algs_name, best_acc)
    except KeyboardInterrupt:
        Log.info('Receive KeyboardInterrupt SIGNAL')
        Log.info('Wait to exit')
        time.sleep(1)
        Log.info('Kill processes in workers')
        PIDManager.SuperEnd.kill_all_process()
        Log.info('Good bye')
        
        
