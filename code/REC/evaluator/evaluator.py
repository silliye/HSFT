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
import re # 导入正则表达式库
from .register import metrics_dict
from .collector import DataStruct
from collections import OrderedDict


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        # ‼️【修改一】: 从配置中解析出所有需要计算的 K 值
        # 例如，如果 metrics 是 ['recall@10', 'ndcg@20']，self.topk 会是 [10, 20]
        self.topk = sorted([int(k) for metric in self.metrics for k in re.findall(r'@(\d+)', metric)])
        if not self.topk:
            raise ValueError("No Top-k value found in metrics config, e.g., 'recall@10'.")

        for metric in self.metrics:
            # 初始化各个指标的计算类
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as `{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}`

        """
        
        # ‼️【修改二】: 在这里补上缺失的核心逻辑
        
        # 1. 从 Collector(dataobject) 中获取模型原始分数
        # 我们假设分数在 Collector 中被注册为 'scores'
        scores = dataobject.get('scores')
        if scores is None:
            raise ValueError("Evaluator requires 'scores' to be registered in the data collector.")

        # 2. 根据分数计算 Top-K 推荐列表
        # 我们需要取配置中最大的 K 值来进行计算，以满足所有指标的需求
        max_k = self.topk[-1]
        _, topk_indices = torch.topk(scores, k=max_k, dim=1)

        # 3. 将 Top-K 结果“注册”回 Collector，供后续使用
        # 这是最关键的一步！
        dataobject.register('rec.topk', topk_indices)

        # 4. 现在可以安全地执行原有的指标计算循环了
        result_dict = OrderedDict()
        for metric in self.metrics:
            # 这里的 calculate_metric 内部会去索要 'rec.topk'
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
            
        return result_dict