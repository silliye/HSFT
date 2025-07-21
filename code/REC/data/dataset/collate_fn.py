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
import numpy as np
from torch.utils.data._utils.collate import default_collate
import re
try:
    from torch._six import string_classes
except:
    string_classes = str

import collections
# [新增] 导入pad_sequence用于处理可变长度的Embedding序列
from torch.nn.utils.rnn import pad_sequence

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

# --- [代码修改] ---
# 这是一个全新的collate_fn，专门为我们的缓存训练流程设计
def cached_train_collate_fn(batch):
    """
    处理CachedEmbeddingTrainDataset输出的batch。
    执行In-batch负采样，并对历史序列进行padding。
    """
    # 1. 过滤掉在Dataset中可能产生的无效样本 (例如因为KeyError)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 2. 分离历史序列和正样本目标
    history_embeddings_list = [item[0] for item in batch]
    positive_target_embeddings = [item[1] for item in batch]

    # 3. 对历史序列进行padding
    #    pad_sequence可以处理一个Tensor列表，并将它们填充到该批次中最长的长度
    #    batch_first=True确保输出的维度是 (batch_size, seq_len, embedding_dim)
    padded_histories = pad_sequence(history_embeddings_list, batch_first=True, padding_value=0)
    
    # 创建attention_mask，用于在模型中忽略padding部分
    # 我们需要知道每个历史序列的真实长度
    history_lengths = torch.tensor([len(hist) for hist in history_embeddings_list])
    attention_mask = torch.arange(padded_histories.size(1))[None, :] < history_lengths[:, None]
    
    # 4. 执行In-batch负采样
    batch_size = len(batch)
    candidate_embeddings = []
    labels = []
    
    # 将所有正样本堆叠成一个Tensor，方便索引
    positive_stack = torch.stack(positive_target_embeddings)

    for i in range(batch_size):
        # 每个样本的候选列表 = 1个自己的正样本 + (batch_size - 1)个别人的正样本(作为负样本)
        
        # 创建一个索引，将第i个样本放在最前面
        indices = torch.roll(torch.arange(batch_size), shifts=-i)
        
        # 根据索引重新排列正样本，形成当前样本的候选列表
        candidates_for_i = positive_stack[indices]
        candidate_embeddings.append(candidates_for_i)
        
        # 创建标签：第一个是1（正样本），其余都是0（负样本）
        label_for_i = torch.zeros(batch_size)
        label_for_i[0] = 1
        labels.append(label_for_i)

    # 5. 将所有数据打包成一个字典
    return {
        # (batch_size, max_seq_len, embedding_dim)
        'history_item_embeddings': padded_histories, 
        # (batch_size, max_seq_len)
        'attention_mask': attention_mask.long(),
        # (batch_size, batch_size, embedding_dim)
        'candidate_item_embeddings': torch.stack(candidate_embeddings),
        # (batch_size, batch_size)
        'labels': torch.stack(labels)
    }

# --- [代码修改] ---
# 这是一个全新的collate_fn，专门为我们的缓存评估流程设计
def cached_eval_collate_fn(batch):
    """
    处理CachedEmbeddingEvalDataset输出的batch。
    对历史序列进行padding，并整理候选列表。
    """
    # 1. 过滤无效样本
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 2. 分离数据
    history_embeddings_list = [item[0] for item in batch]
    candidate_embeddings_list = [item[1] for item in batch]
    target_indices = [item[2] for item in batch] # 实际上都是0

    # 3. 对历史序列进行padding
    padded_histories = pad_sequence(history_embeddings_list, batch_first=True, padding_value=0)
    history_lengths = torch.tensor([len(hist) for hist in history_embeddings_list])
    attention_mask = torch.arange(padded_histories.size(1))[None, :] < history_lengths[:, None]

    # 4. 堆叠候选列表
    #    因为我们在Dataset中保证了每个候选列表长度相同 (1正+N负)，所以可以直接stack
    candidate_embeddings = torch.stack(candidate_embeddings_list)

    # 5. 打包成字典
    return {
        'history_item_embeddings': padded_histories,
        'attention_mask': attention_mask.long(),
        'candidate_item_embeddings': candidate_embeddings,
        'target_index': torch.tensor(target_indices) # (batch_size)
    }
# --------------------


def customize_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return batch


def seq_eval_collate(batch):
    item_seq = []
    item_target = []
    time_seq = []

    history_i = []

    for item in batch:
        history_i.append(item[0])
        item_seq.append(item[1])
        item_target.append(item[2])
        time_seq.append(item[3])

    history_u = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_i)])
    history_i = torch.cat(history_i)

    item_seq = torch.tensor(item_seq)  # [batch, len]
    item_target = torch.tensor(item_target)  # [batch]
    time_seq = torch.tensor(time_seq)  # [batch]
    positive_u = torch.arange(item_seq.shape[0])  # [batch]

    # return item_seq, None, positive_u, item_target
    return item_seq, time_seq, (history_u, history_i), positive_u, item_target


def customize_rmpad_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        output = {}
        for key in elem:
            if any(['_input_ids' in key, '_cu_input_lens' in key, '_position_ids' in key]):
                output[key] = torch.concat([d[key] for d in batch], dim=0)
            else:
                output[key] = customize_collate([d[key] for d in batch])
        return output
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        return batch
