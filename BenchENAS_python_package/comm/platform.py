import paramiko


def linux_win(ssh_name, ssh_pwd, ip, port):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, ssh_name, ssh_pwd)
    cmd_linux = 'uname -a'
    _, stdout, stderr = ssh.exec_command(cmd_linux)
    # stdout = stdout.read().decode()
    stderr = stderr.read().decode('gbk')

    if len(stderr) > 0:
        cmd_win = 'ver'
        _, stdout, stderr = ssh.exec_command(cmd_win)
        stderr = stderr.read().decode('gbk')
        if len(stderr) > 0:
            system = False
        else:
            system = 'windows'
    else:
        system = 'linux'
    ssh.close()

    return system


def run_cmd_list(ssh_name, ssh_pwd, ip, port, cmd_list):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, ssh_name, ssh_pwd)
    _, _, sys_err = ssh.exec_command('ver')
    sys_err = sys_err.read().decode()
    for cmd_ in cmd_list:
        _, _, stderr = ssh.exec_command(cmd_)
        if len(sys_err) > 0:
            stderr = stderr.read().decode()
        else:
            stderr = stderr.read().decode('gbk')
        if len(stderr) > 0:
            ssh.close()
            return stderr
        else:
            pass
    ssh.close()
    return False


def run_cmd(ssh_name, ssh_pwd, ip, port, _cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, ssh_name, ssh_pwd)
    _, stdout, stderr = ssh.exec_command(_cmd)
    _, _, sys_err = ssh.exec_command('ver')
    sys_err = sys_err.read().decode()
    if len(sys_err) > 0:
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
    else:
        stdout = stdout.read().decode('gbk')
        stderr = stderr.read().decode('gbk')
    ssh.close()
    return stdout, stderr
