import os
import platform
import time
import sys
import importlib
import glob
import subprocess
import selectors

import multiprocess
import paramiko

from comm.platform import linux_win, run_cmd_list, run_cmd
from compute import Config_ini
from compute.log import Log


def get_local_path():
    """
        :return:Get the absolute path of the calling package
    """
    _path = os.getcwd()
    return _path


def get_transfer_local_path():
    """
        :return:Get the absolute path of the package
    """
    _path = os.path.dirname(os.path.dirname(__file__))
    return _path


def get_algo_name():
    """
        :return:Get the name of the currently running algorithm
    """
    alg_name = Config_ini.alg_name
    return alg_name


def get_gen_number():
    """
        :return:Get the algebra of NAS iterations
    """
    max_gen = Config_ini.max_gen
    return int(max_gen)


def get_pop_siz():
    """
        :return:Get population size
    """
    pop_size = Config_ini.pop_size
    return int(pop_size)


def get_exe_path():
    exe_path = Config_ini.exe_path
    return exe_path


def get_algo_local_dir():
    """
        :return:Get the corresponding folder under the running algorithm runtime
    """
    top_dir = get_local_path()
    alg_name = Config_ini.alg_name

    local_dir = os.path.join(top_dir, 'runtime', alg_name)
    if not os.path.exists(os.path.dirname(local_dir)):
        os.mkdir(os.path.dirname(local_dir))
    return local_dir


def get_population_dir():
    """
        :return:Get the populations folder in the corresponding folder under the running algorithm runtime and create it
    """
    pop_dir = os.path.join(get_algo_local_dir(), 'populations')
    if not os.path.exists(pop_dir):
        os.makedirs(pop_dir)

    return pop_dir


def get_top_dest_dir():
    """
        :return:Get the path of the algorithm under the server root path
    """
    alg_name = Config_ini.alg_name
    tdd = os.path.join('~', alg_name)
    return tdd


def get_train_ini_path():
    """
        :return:Get the absolute path of train.ini
    """
    return os.path.join(get_local_path(), 'train', 'train.ini')


def get_global_ini_path():
    """
        :return:Get the absolute path of global.ini
    """
    return os.path.join(get_local_path(), 'global.ini')


def exec_cmd_remote(_cmd, need_response=True):
    p = subprocess.Popen(_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout_str = None
    stderr_str = None

    if need_response:
        sel = selectors.DefaultSelector()
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)
        stdout_ = None
        stderr_ = None
        for key, _ in sel.select():
            data = key.fileobj.readlines()

            if key.fileobj is p.stdout:
                stdout_ = data
            else:
                stderr_ = data

        if stdout_ is not None and len(stdout_) > 0:
            stdout_str = ''.join([_.decode('utf-8') for _ in stdout_])

        if stderr_ is not None and len(stderr_) > 0:
            stderr_str = ''.join([_.decode('utf-8') for _ in stderr_])

    return stdout_str, stderr_str


def detect_file_exit(ssh_name, ssh_pwd, ip, port, file_name):
    transport = paramiko.Transport((ip, port))
    transport.connect(username=ssh_name, password=ssh_pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir('.')
    try:
        sftp.stat(file_name)
        return True
    except:
        return False


def init_work_dir(ssh_name, ssh_password, ip, port):
    Log.debug('Start to init the work directory in each worker')
    alg_name = get_algo_name()
    cmd_ = list()

    if detect_file_exit(ssh_name, ssh_password, ip, port, alg_name):
        system_ver = linux_win(ssh_name, ssh_password, ip, port)
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        if system_ver == 'linux':
            _bak_cmd = 'mv \'%s\' \'%s_bak_%s\'' % (alg_name, alg_name, time_str)
            cmd_.append(_bak_cmd)
        elif system_ver == 'windows':
            _bak_cmd = 'ren %s %s_bak_%s' % (alg_name, alg_name, time_str)
            cmd_.append(_bak_cmd)
        else:
            Log.debug('Current system %s is not windows or linux!' % ip)

    _mk_cmd = 'mkdir %s' % alg_name
    cmd_.append(_mk_cmd)
    for cmd in cmd_:
        Log.info('Execute the cmd: %s' % cmd)
    stderr_ = run_cmd_list(ssh_name, ssh_password, ip, port, cmd_)
    if stderr_:
        Log.debug('Stderr: %s' % stderr_)


def init_work_dir_on_all_workers():
    Log.info('Init the work directories on each worker')
    gpu_info = Config_ini.gpu_info
    ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100']
    for sec in gpu_info.keys():
        worker_name = gpu_info[sec]['worker_name']
        worker_ip = gpu_info[sec]['worker_ip']
        ssh_name = gpu_info[sec]['ssh_name']
        ssh_password = gpu_info[sec]['ssh_password']
        port = gpu_info[sec]['port']
        init_work_dir(ssh_name, ssh_password, worker_ip, port)
        transfer_training_files(ssh_name, ssh_password, worker_ip, port)
        if Config_ini.dataset not in ls_dataset:
            transfer_dataset_image(ssh_name, ssh_password, worker_ip, port, Config_ini.data_dir)


def makedirs(sftp, dir_path):
    Log.info('Execute the operation: mkdir %s' % dir_path)
    try:
        sftp.stat(dir_path)
    except:
        sftp.mkdir(dir_path)


def exec_python(ssh_name, ssh_pwd, ip, port, py_file, args, python_exec):
    top_dir = get_top_dest_dir()
    py_file = os.path.join(top_dir, py_file).replace('~', '.').replace('\\', '/')
    # compute.log输出
    Log.info('Execute the remote python file [(%s)%s]' % (ip, py_file))
    _exec_cmd = '%s %s %s' % (python_exec, py_file,
                              ' '.join([' '.join([k, v]) for k, v in
                                        args.items()]))
    Log.info('Execute the cmd: %s' % _exec_cmd)
    p = multiprocess.Process(target=run_cmd, args=(ssh_name, ssh_pwd, ip, port, _exec_cmd))
    p.start()


def transfer_file_relative(ssh_name, ssh_pwd, ip, port, source, dest):
    """Use relative path to transfer file, both source and dest are relative path
    """

    top_dir = get_top_dest_dir()
    full_path_dest = os.path.join(top_dir, dest).replace('~', '.')
    full_path_dest = full_path_dest.replace('\\', '/')
    full_path_source = os.path.join(get_local_path(), source).replace('\\', '/')
    transport = paramiko.Transport((ip, port))
    transport.connect(username=ssh_name, password=ssh_pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    # full_path_source = full_path_source.replace(' ','\\\\ ')
    makedirs(sftp, os.path.dirname(full_path_dest))
    try:
        Log.info('Execute the operation: put %s to %s' % (full_path_source, full_path_dest))
        sftp.put(full_path_source, full_path_dest)
        Log.info('Transfer file successfully...')
    except Exception as e:
        Log.info('Transfer file failed....')
        Log.debug(e)

    sftp.close()


def sftp_makedirs(sftp_sess, dir_path):
    cwd_bak = sftp_sess.getcwd()
    dir_split = [dir_path]
    while os.path.dirname(dir_path) != '' and os.path.dirname(dir_path) != '/':
        dir_split = [os.path.dirname(dir_path)] + dir_split
        dir_path = dir_split[0]

    for dir_ in dir_split:
        try:
            # exists
            sftp_sess.stat(dir_)
        except:
            # absent
            sftp_sess.mkdir(dir_)

    sftp_sess.chdir(cwd_bak)


def sftp_transfer(sftp_sess, src_path, dst_path):
    sftp_makedirs(sftp_sess, os.path.dirname(dst_path))
    sftp_sess.put(src_path, dst_path)


def transfer_training_files(ssh_name, ssh_password, worker_ip, port):
    training_file_dep = [(v, v) for _, v in get_training_file_dependences().items()]
    transport = paramiko.Transport((worker_ip, port))
    transport.connect(username=ssh_name, password=ssh_password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir('.')
    root_dir = sftp.getcwd()

    sub_file = os.path.dirname(os.path.dirname(__file__))
    sub_file = os.path.join(sub_file, 'runtime/README.MD').replace('\\', '/')
    training_file_dep = training_file_dep + [(sub_file, 'runtime/README.MD')]

    top_dir = get_top_dest_dir()
    for src, dst in training_file_dep:
        full_path_source = os.path.join(get_transfer_local_path(), src)
        full_path_dest = os.path.join(top_dir, dst).replace('~', root_dir).replace('\\', '/')

        if full_path_dest.endswith('training.py'):
            full_path_dest = os.path.join(os.path.dirname(os.path.dirname(full_path_dest)), 'training.py').replace('\\',
                                                                                                                   '/')
        Log.debug('Start to sftp: `%s` ==> `%s`' % (full_path_source, full_path_dest))
        sftp_transfer(sftp, full_path_source, full_path_dest)

    transport.close()


def transfer_dataset_image(ssh_name, ssh_password, worker_ip, port, source):
    transport = paramiko.Transport((worker_ip, port))
    transport.connect(username=ssh_name, password=ssh_password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir('.')
    root_dir = sftp.getcwd()
    source = source.replace('\\', '/')
    dset_name = source.split('/')[-1]
    try:
        sftp.stat(dset_name)
    except:
        sftp.mkdir(dset_name)

    for root, subdir, files in os.walk(source):
        for dir in subdir:
            r_d = source.split("/")
            local_subdir = os.path.join(root, dir).replace('\\', '/')
            l_d = local_subdir.split("/")
            r_m = l_d[len(r_d):]
            r_m = "/".join(r_m)
            remote_subdir = os.path.join(dset_name, r_m).replace('\\', '/')
            try:
                sftp.stat(remote_subdir)
            except:
                sftp.mkdir(remote_subdir)
        for file in files:
            local_dir_path = os.path.join(root, file).replace('\\', '/')
            l_d_p = local_dir_path.split("/")
            r_d_p = l_d_p[len(r_d):]
            r_d_p = "/".join(r_d_p)
            remote_dir_path = os.path.join(dset_name, r_d_p).replace('\\', '/')
            Log.info('Start to sftp dataset: `%s` ==> `%s`' %
                     (local_dir_path,
                      os.path.join(root_dir, remote_dir_path).replace('\\', '/')))
            try:
                sftp.stat(remote_dir_path)
            except:
                sftp.put(local_dir_path, remote_dir_path)

    transport.close()


def get_dependences_by_module_name(module_name):
    import multiprocessing
    with multiprocessing.Pool(1) as p:
        res = p.map(__help_func, (module_name,))[0]

    return res


def get_training_file_dependences():
    f_list = list(filter(lambda x: not x.startswith(os.path.join(get_transfer_local_path(), 'runtime')) and
                                   not x.startswith(os.path.join(get_transfer_local_path(), 'venv')) and
                                   not x.startswith(os.path.join(get_transfer_local_path(), '__pycache__')),
                         glob.iglob(os.path.join(get_transfer_local_path(), '**/*.py'),
                                    recursive=True))) + \
             list(filter(lambda x: not x.startswith(os.path.join(get_transfer_local_path(), 'runtime')) and
                                   not x.startswith(os.path.join(get_transfer_local_path(), '__pycache__')),
                         glob.iglob(os.path.join(get_transfer_local_path(), '**/*.ini'),
                                    recursive=True)))

    if platform.system() == 'Windows':
        res = {
            _.replace(get_transfer_local_path() + '\\', ''):
                _.replace(get_transfer_local_path() + '\\', '')
            for _ in f_list}
    else:
        res = {
            _.replace(get_transfer_local_path() + '/', ''):
                _.replace(get_transfer_local_path() + '/', '')
            for _ in f_list}

    return res


def get_all_edl_modules():
    """Get name and relative path of the modules in edl project
    """
    res = {}
    for k, v in sys.modules.items():
        if hasattr(v, '__file__'):
            if v is not None:
                try:
                    if v.__file__ and 'site-packages' in getattr(v, '__file__'):
                        pass
                    else:
                        project_dir = get_local_path()
                        if v.__file__ and v.__file__.startswith(project_dir):
                            res[k] = v.__file__.replace(project_dir + '/', '')
                except Exception:
                    import pdb
                    pdb.set_trace()
        else:
            pass

    return res


def __help_func(module_name):
    importlib.import_module('.', module_name)

    res = get_all_edl_modules()

    return res


if __name__ == '__main__':
    print(get_training_file_dependences())
