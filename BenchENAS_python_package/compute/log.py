import logging
import sys
import os


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("PlatENAS_Compute")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_path = os.path.join(os.getcwd(), 'runtime', 'compute.log')
            if not os.path.exists(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
                file = open(file_path, 'w+')
                file.close()
            else:
                if not os.path.exists(file_path):
                    file = open(file_path, 'w+')
                    file.close()
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
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
