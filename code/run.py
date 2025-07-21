# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse
# 导入 safetensors 的加载函数，以备将来使用，但在此逻辑中不再直接调用
from safetensors.torch import load_file

def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s

def run_loop(local_rank, config_file=None, saved=True, extra_args=[], is_distributed=False):
    # configurations initialization
    config = Config(config_file_list=config_file)

    device = torch.device("cuda", local_rank)
    config['device'] = device

    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")

    # get model and data
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")

    # 模型已经在这里通过 get_model 加载了 ItemLLM 和 UserLLM 的权重
    model = get_model(config['model'])(config, dataload).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info(model)

    # --- 核心修改部分 ---
    if config['val_only']:
        # 在“仅验证”模式下，我们假设模型已加载预训练权重，
        # 因此直接跳过训练，开始评估。
        logger.info("'val_only' is True. Skipping training and proceeding directly to evaluation.")
        
        # 使用当前已加载的模型进行评估，无需加载额外的checkpoint
        test_result = trainer.evaluate(test_loader, load_best_model=False, show_progress=config['show_progress'], init_model=True)
        
        logger.info(set_color('Test result', 'yellow') + f': {test_result}')
    else:
        # 正常训练流程
        logger.info("Starting training process...")
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended. ' + set_color('Best valid result', 'yellow') + f': {best_valid_result}')

        # 使用训练后得到的最佳模型进行最终测试
        logger.info("Starting final evaluation on test set...")
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('Best valid result', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('Test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str, help="Path to config file(s)")
    args, extra_args = parser.parse_known_args()
    config_file = args.config_file

    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        is_distributed = True
    else:
        is_distributed = False

    if is_distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        print(f"Running in distributed mode. Rank: {local_rank}")
    else:
        local_rank = 0
        print("Running in single-card mode.")

    run_loop(
        local_rank=local_rank,
        config_file=config_file,
        extra_args=extra_args,
        is_distributed=is_distributed
    )
