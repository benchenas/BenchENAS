import logging
import sys
import os

class Log(object):
    _logger = None
    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("PlatENAS")  #获取 记录器 name = PlatENAS
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s') #  设置输出格式  时间 日志级别  信息
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.log')
            file_handler = logging.FileHandler(file_path)  #将日志输入到目录的目录的磁盘文件上
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)
        
    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warn(_str)
        
    @classmethod
    def debug(cls, _str):
        cls.__get_logger().debug(_str)