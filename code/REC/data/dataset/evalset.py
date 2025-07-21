# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import numpy as np
import datetime
import pytz
import random # [新增] 导入random库用于负采样

# --- [代码修改] ---
# 这是一个全新的Dataset类，专门用于处理预计算好的Embedding缓存
# 它将用于验证和测试
class CachedEmbeddingEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.config = config
        self.dataload = dataload
        self.phase = phase

        # 1. 从 dataload 对象获取缓存和至关重要的“翻译词典”
        self.item_embeddings = dataload.item_embeddings_cache
        self.id2token = dataload.id2token['item_id'] # 整数ID -> 原始字符串ID 的映射
        self.token2id = dataload.token2id['item_id'] # 原始字符串ID -> 整数ID 的映射
        print(self.id2token)
        if self.item_embeddings is None:
            raise ValueError("错误：CachedEmbeddingEvalDataset需要预计算的Embedding缓存，但未提供。")

        # 2. 获取用户序列（内部为程序运行时的整数ID）
        self.user_seq = list(dataload.user_seq.values())
        self.length = len(self.user_seq)
        
        # 3. 获取所有物品的“原始字符串ID”列表，用于负采样
        self.all_item_original_ids = list(self.item_embeddings.keys())
        
        # 4. 读取配置
        self.max_item_list_length = config.get('MAX_ITEM_LIST_LENGTH_TEST', config.get('MAX_ITEM_LIST_LENGTH', 50))
        self.neg_sample_num = config.get('eval_neg_sample_num', 99)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # a. 根据phase（'valid'或'test'）确定切分点和用户序列
        last_num = 2 if self.phase == 'valid' else 1
        user_seq_ids = self.user_seq[index] # 这是整数ID序列
        
        # 序列太短，无法构成评估样本，直接跳过
        if len(user_seq_ids) < last_num + 1:
            return None

        # 切分出历史和目标（依然是整数ID）
        history_ids = user_seq_ids[:-last_num]
        target_id = user_seq_ids[-last_num]
        history_ids = history_ids[-self.max_item_list_length:]
        
        # b. 构建候选ID列表 (1个正样本 + N个负样本)
        candidate_ids = [target_id] # 列表内始终存储整数ID
        
        try:
            # 【修复负采样逻辑】将用户交互过的物品ID也转换成原始字符串格式，用于去重
            seen_items_original = {self.id2token[iid] for iid in user_seq_ids}
        except IndexError:
            # 如果user_seq_ids中的某个ID超出了映射表范围，说明数据有问题，跳过该样本
            return None

        # 进行负采样
        neg_count = 0
        max_tries = self.neg_sample_num * 10 # 设置最大尝试次数，防止死循环
        current_tries = 0
        while neg_count < self.neg_sample_num and current_tries < max_tries:
            # 从原始字符串ID列表中随机抽取一个
            neg_id_original = random.choice(self.all_item_original_ids)
            current_tries += 1
            if neg_id_original not in seen_items_original:
                # 检查这个字符串ID是否存在于反向映射表中
                if neg_id_original in self.token2id:
                    # 将采样到的原始字符串ID，翻译回整数ID，并添加到候选列表中
                    candidate_ids.append(self.token2id[neg_id_original])
                    seen_items_original.add(neg_id_original) # 避免重复采样
                    neg_count += 1
        
        # 如果负采样数量不足，说明物品太少或用户历史太长，跳过此样本
        if neg_count < self.neg_sample_num:
            return None

        # c. 【核心修复】将整数ID列表统一转换为缓存字典所需的字符串ID格式，并查询Embedding
        try:
            # 使用 dataload 中存储的映射表进行“翻译”
            history_original_ids = [self.id2token[item_id] for item_id in history_ids]
            candidate_original_ids = [self.id2token[item_id] for item_id in candidate_ids]
            
            # 使用翻译后的字符串ID去查询缓存字典
            history_embeddings = [self.item_embeddings[original_id] for original_id in history_original_ids]
            candidate_embeddings = [self.item_embeddings[original_id] for original_id in candidate_original_ids]
            
            # 正样本在候选列表中的索引位置永远是0
            target_index = 0 

            # 将Embedding列表堆叠成Tensor并返回
            return (
                torch.stack(history_embeddings), 
                torch.stack(candidate_embeddings),
                target_index
            )
        except (KeyError, IndexError) as e:
            # KeyError: 缓存中找不到翻译后的字符串ID
            # IndexError: 某个整数ID超出了id2token映射表的范围
            # 无论哪种情况，都说明数据不一致，记录警告并跳过该样本
            logger.warning(f"警告: ID '{e}' 无法被正确处理。该评估样本将被跳过。")
            return None
# --------------------

class SeqEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH_TEST'] if config['MAX_ITEM_LIST_LENGTH_TEST'] else config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values())
        self.time_seq = list(dataload.time_seq.values())
        self.use_time = config['use_time']
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return sequence

    def _padding_time_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return vq_time

    def __getitem__(self, index):
        last_num = 2 if self.phase == 'valid' else 1
        history_seq = self.user_seq[index][:-last_num]
        item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
        item_target = self.user_seq[index][-last_num]
        if self.use_time:
            history_time_seq = self.time_seq[index][:-last_num]
        else:
            history_time_seq = []
        time_seq = self._padding_time_sequence(history_time_seq, self.max_item_list_length)

        return torch.tensor(history_seq), item_seq, item_target, time_seq  # , item_length
