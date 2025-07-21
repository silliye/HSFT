# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import argparse

# --- [代码修改] ---
# 新增一个辅助函数，用于将命令行中的字符串'True'/'False'转为布尔值
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# --------------------

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["OMP_NUM_THREADS"] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+')
    
    # 添加三个新的命令行参数，用于控制我们的新流程
    parser.add_argument('--use_item_cache', type=str2bool, default=False, help='whether to use pre-computed item embeddings')
    parser.add_argument('--item_cache_path', type=str, default='dataset/pixel200k_item_embeddings.pt', help='path to the item embedding cache file')
    parser.add_argument('--use_qlora', type=str2bool, default=False, help='whether to use QLoRA for fine-tuning')

    args, unknown_args = parser.parse_known_args()
    config_file = args.config_file

    # 将新添加的参数也拼接成字符串，传递给run.py
    new_args_str = f" --use_item_cache {args.use_item_cache} --item_cache_path {args.item_cache_path} --use_qlora {args.use_qlora}"

    # --- [核心修改] ---
    # 彻底移除不存在的 "../TORCHRUN"
    # 直接用python来执行run.py，这是标准的做法
    if len(config_file) == 2:
        run_yaml = f"python code/run.py --config_file {config_file[0]} {config_file[1]} {' '.join(unknown_args)} {new_args_str}"
    elif len(config_file) == 1:
        run_yaml = f"python code/run.py --config_file {config_file[0]} {' '.join(unknown_args)} {new_args_str}"
    # --------------------
    print(run_yaml)

    os.system(run_yaml)
