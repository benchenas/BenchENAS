import sqlite3
import os
import time
from compute.log import Log
from compute.config import AlgorithmConfig
from compute.file import get_local_path


def get_db_name():
    return 'PlatENAS'


def get_db_path():
    file_path = os.path.join(get_local_path(), 'runtime', '%s.db' % (get_db_name()))
    return file_path


def init_db():
    Log.info('Initialize the database')
    try:
        conn = sqlite3.connect(get_db_path())
        rs = check_table(conn, 'gpu_list')
        if len(rs) == 0:
            Log.debug('Tables do not exist, creating them first')
            _create_table_sql = '''CREATE TABLE %s
                               (id  integer PRIMARY KEY autoincrement,
                               alg_name              VARCHAR(250),
                               worker                VARCHAR(250),		     
                               ssh_name              VARCHAR(250),
                               ssh_password          VARCHAR(250),
                               gpu_id                int(8),
                               status                int(8),
                               remark                text,
                               time                  VARCHAR(100));'''

            Log.debug('Init the gpu_list table ...')
            conn.execute(_create_table_sql % ('gpu_list'))
            Log.debug('Init the gpu_arxiv_list table ...')
            conn.execute(_create_table_sql % ('gpu_arxiv_list'))

            # init the gpu_use table
            _create_table_sql = '''CREATE TABLE gpu_use
                                   (id integer PRIMARY KEY autoincrement,
                                   alg_name               VARCHAR(250),
                                   worker                 VARCHAR(250),
                                   gpu_id                 VARint(8),
                                   status                 VARCHAR(5),
                                   script_name            VARCHAR(250),
                                   time                   VARCHAR(100));'''
            Log.debug('Init the gpu_use table ...')
            conn.execute(_create_table_sql)

            conn.commit()
        else:
            Log.debug('Table gpu_list already exists')

        conn.close()
    except BaseException as e:
        Log.warn('Errors when initializing the database [%s]' % (str(e)))


def check_table(conn, table_name):
    sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'%s\';' % (table_name)
    cu = conn.cursor()
    cu.execute(sql)
    rs = cu.fetchall()
    return rs


def add_info(gpu_info, info):
    '''
    Two tables will be operated, gpu_list and gpu_arxiv_list
    all the gpu infomation should be added to gpu_arxiv_list
    while only the available gpu list is added to gpu_list, because operating this table, a delete operation should be given
    '''
    conn = sqlite3.connect(get_db_path())
    alg_config = AlgorithmConfig()
    cu = conn.cursor()
    alg_name = alg_config.read_ini_file('name')
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    sql_gpu_list_del = 'delete from gpu_list where alg_name = \'%s\'' % (alg_name)
    Log.debug('Execute sql: %s' % (sql_gpu_list_del))
    cu.execute(sql_gpu_list_del)
    for each_one in info:

        worker = each_one['worker_ip']
        ssh_name = gpu_info[each_one['worker_ip']]['ssh_name']
        ssh_password = gpu_info[each_one['worker_ip']]['ssh_password']
        gpu_id = each_one['gpu_id']
        status = each_one['status']
        remark = each_one['remark']

        if status == 0:
            sql_gpu_list = 'INSERT INTO gpu_list(alg_name, worker, ssh_name, ssh_password, gpu_id, status, remark, time) values (\'%s\', \'%s\', \'%s\', \'%s\', %d, %d, \'%s\', \'%s\')' % (
                alg_name, worker, ssh_name, ssh_password, gpu_id, status, remark, time_str
            )
            Log.debug('Execute sql: %s' % (sql_gpu_list))
            cu.execute(sql_gpu_list)

        sql_gpu_archiv_list = 'INSERT INTO gpu_arxiv_list(alg_name, worker, ssh_name, ssh_password, gpu_id, status, remark, time) values (\'%s\', \'%s\', \'%s\', \'%s\', %d, %d, \'%s\', \'%s\')' % (
            alg_name, worker, ssh_name, ssh_password, gpu_id, status, remark, time_str
        )
        Log.debug('Execute sql: %s' % (sql_gpu_archiv_list))
        cu.execute(sql_gpu_archiv_list)
    conn.commit()
    conn.close()


def get_available_gpus():
    alg_config = AlgorithmConfig()
    alg_name = alg_config.read_ini_file('name')
    sql = 'select id, worker as worker_ip, gpu_id, ssh_name, ssh_password from gpu_list where alg_name=\'%s\'' % (
        alg_name)
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    rs = cu.fetchall()
    conn.close()
    return rs


def confirmed_used_gpu(ids):
    sql = 'delete from gpu_list where id in (%s)' % (','.join(ids))
    Log.debug('Execute sql: %s' % (sql))
    conn = sqlite3.connect(get_db_path())
    cu = conn.cursor()
    cu.execute(sql)
    conn.commit()
    conn.close()
