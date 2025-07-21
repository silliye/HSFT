# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist


# 文件: trainset.py (或您定义Dataset的文件)

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist

class CachedEmbeddingTrainDataset(Dataset):
    """
    用于训练的Dataset，处理预计算的Embedding缓存。
    [最终修复版]
    """
    def __init__(self, config, dataload):
        self.config = config
        self.dataload = dataload
        
        # 1. 从 dataload 对象获取缓存和“翻译词典”
        self.item_embeddings = dataload.item_embeddings_cache
        self.id2token = dataload.id2token['item_id'] # 整数ID -> 原始字符串ID 的映射

        if self.item_embeddings is None:
            raise ValueError("错误：CachedEmbeddingTrainDataset需要预计算的Embedding缓存，但未提供。")

        # 2. 获取训练序列（内部为程序运行时的整数ID）
        self.train_seq = dataload.train_feat['item_seq']
        self.length = len(self.train_seq)
        
        # 3. 读取配置
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH'] + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # a. 获取一个用户的完整交互整数ID序列
        item_seq_ids = self.train_seq[index]
        
        # 序列太短，无法构成(历史, 目标)训练样本
        if len(item_seq_ids) < 2:
            return None
        
        # b. 随机选择一个切分点，生成(历史, 目标)对
        split_point = random.randint(1, len(item_seq_ids) - 1)
        history_ids = item_seq_ids[:split_point]
        target_id = item_seq_ids[split_point]
        
        # 截断长序列
        history_ids = history_ids[-self.max_seq_length+1:]

        # c. 将整数ID翻译回原始的字符串ID，再去查询Embedding
        try:
            history_original_ids = [self.id2token[item_id] for item_id in history_ids]
            target_original_id = self.id2token[target_id]

            history_embeddings = [self.item_embeddings[original_id] for original_id in history_original_ids]
            target_embedding = self.item_embeddings[target_original_id]
            
            # ‼️ 【核心修复】: 在返回前，将历史Embedding列表堆叠成一个单独的Tensor
            # 这样输出的样本格式就是 (Tensor, Tensor)，下游的collate_fn可以正确处理
            if not history_embeddings: # 处理历史为空的边界情况
                history_tensor = torch.empty(0, target_embedding.shape[0], dtype=target_embedding.dtype)
            else:
                history_tensor = torch.stack(history_embeddings)

            return (history_tensor, target_embedding)

        except (KeyError, IndexError) as e:
            logger.warning(f"警告：ID '{e}' 无法被正确处理。该训练样本将被跳过。")
            return None

# 数据形式为 [[user_seq], [neg_item_seq]] , [mask]
class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']

        self.length = len(self.train_seq)

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        logger.info(f"Use random sample {self.random_sample} for mask id")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample(item_seq))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        return torch.as_tensor(item_seq, dtype=torch.int64), torch.as_tensor(neg_item, dtype=torch.int64), torch.as_tensor(masked_index, dtype=torch.int64)

    def __getitem__(self, index):
        # 最长长度为maxlen+1, 及若max_len是5
        # 则存在    1,2,3,4,5,6序列,
        # pos       2,3,4,5,6
        # neg       0,8,9,7,9,8
        # mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)

        return item_seq, neg_item, masked_index


