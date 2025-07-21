# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import logging
import os
import sys
import colorlog
import re
import torch
from REC.utils.utils import get_local_time, ensure_dir
from colorama import init

log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """

    init(autoreset=True)
    LOGROOT = config['checkpoint_dir'] + '/' if config['checkpoint_dir'] else './log/'
    dir_name = os.path.dirname(LOGROOT)

    # --- 核心修改部分 ---
    # 1. 安全地检查分布式环境是否已初始化
    if torch.distributed.is_initialized():
        # 如果是分布式模式, 则获取真实的 rank
        rank = torch.distributed.get_rank()
    else:
        # 如果是单卡模式, 我们就默认 rank 为 0
        rank = 0

    # 2. 只有主进程 (rank 0) 才创建目录
    #    这个逻辑在两种模式下都有效
    if rank == 0:
        ensure_dir(dir_name)
        model_name = os.path.join(dir_name, config['model'])
        ensure_dir(model_name)

    # 3. 只有在分布式模式下才需要同步进程
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # --- 修改结束 ---

    logfilename = '{}/{}.log'.format(config['model'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)
    if config['log_path']:
        logfilepath = os.path.join(LOGROOT, config['log_path'])

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    # 这个逻辑现在也是安全的，因为 rank 在两种模式下都有正确的值
    logging.basicConfig(level=level if rank in [-1, 0] else logging.WARN, handlers=[sh, fh])
