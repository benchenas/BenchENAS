import os.path


def read_result(alg_name):
    file_name = os.getcwd()
    result_path = os.path.join(file_name, 'runtime', alg_name, 'populations', 'results.txt').replace('\\', '/')
    best_acc = 0.0
    with open(result_path, 'r') as f:
        line = f.readline().strip()
        while line:
            acc = line.split('=')[1]
            if float(acc) > best_acc:
                best_acc = float(acc)
            line = f.readline().strip()
        f.close()
    return best_acc


def write_result(alg_name, best_acc):
    file_name = os.getcwd()
    result_path = os.path.join(file_name, 'runtime', 'results.txt').replace('\\', '/')
    if not os.path.exists(result_path):
        file = open(result_path, 'w')
        file.close()
    fd = open(result_path, "a")
    res = f'\n{alg_name}={best_acc}'
    fd.write(res)
    fd.close()

