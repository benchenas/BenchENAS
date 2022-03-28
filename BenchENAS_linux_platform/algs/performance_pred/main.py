import time, os
from algs.performance_pred.utils import StatusUpdateTool, Log, GenerateCNNToFile
from algs.performance_pred.gen_pycode import GenPyModel
from compute.gpu import gpus_all_available
from compute.file import get_algo_local_dir
from compute.process import dispatch_to_do

class Run():
    
    def __init__(self):
        self.pop_dir = os.path.join(get_algo_local_dir(),'populations')

    def get_all_cnns(self):
        file_name = os.path.join(self.pop_dir, 'networks.txt')
        if not os.path.exists(file_name):
            exit('file: %s does not exist'%(file_name))
            
        all_cnns = []
        all_cnn_uuids = {}
        f = open(file_name)
        lines = f.readlines()
        for l in lines:
            if l.startswith('id'):
                _map = {}
                array1 = l.split(',')
                for str1 in array1:
                    array2 = str1.strip().split(':')
                    _map[array2[0].strip()] = array2[1].strip()
                all_cnns.append(_map['id'])
                all_cnn_uuids[_map['id']] = _map['uuid']    
        return all_cnns, all_cnn_uuids

    def get_evaluated_cnns(self):
        evaluated_cnns = []
        file_name = os.path.join(self.pop_dir, 'results.txt')
        if not os.path.exists(file_name):
            return evaluated_cnns
        else:
            f = open(file_name)
            lines = f.readlines()
            for l in lines:
                _id, _ = l.strip().split('=')
                evaluated_cnns.append(_id)
        return evaluated_cnns
            
    
    def find_all_not_evaluated_cnns(self):
        all_cnns, all_cnn_uuids = self.get_all_cnns()
        evaluated_cnns = self.get_evaluated_cnns()
        non_evaluated_cnns = []
        for cnn in all_cnns:
            if cnn not in evaluated_cnns:
                non_evaluated_cnns.append(cnn)
        Log.info('%d CNNs have not evaluated'%(len(non_evaluated_cnns)))
        return non_evaluated_cnns, all_cnn_uuids
        

    
    def do(self):
        # create dir in `runtime/${ALGO_NAME}`
        alg_dir = get_algo_local_dir()
        if not os.path.exists(alg_dir):
            os.mkdir(alg_dir)
        
        
        if StatusUpdateTool.is_evolution_running():
            Log.info('Start to run from the last termination point')
        else:
            Log.info('Begin to run')
            StatusUpdateTool.begin_evolution()
            # generate cnn architectures and then save them to the text file
            generateCNNtoFile = GenerateCNNToFile(StatusUpdateTool.get_sample_number())
            generateCNNtoFile.do()
            # generate CNN scripts
            pyGen = GenPyModel()
            #pyGen.convert()
            pyGen.convert()
        # find all the cnns that have not been evaluated
        non_evaluated_cnns, all_cnn_uuids = self.find_all_not_evaluated_cnns()
        if len(non_evaluated_cnns) > 0:
            for cnn in non_evaluated_cnns:
                Log.info('Begin to evaluate %s'%(cnn))
                dispatch_to_do(cnn, all_cnn_uuids[cnn])
         
        all_have_been_evaluated = False
        while all_have_been_evaluated is not True:
            #print('All have been evaluated flag ', all_have_been_evaluated)
            time.sleep(120)
            all_have_been_evaluated = gpus_all_available()
         
        Log.info('All CNNs have been evaluated, please to check GPU to ensure all GPUs have been released')
        StatusUpdateTool.end_evolution()
            
            
            
if __name__ == '__main__':
    r = Run()
    r.do()
    
        
        