# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from logging import getLogger

# [新增] 导入 QLoRA 所需的库
from peft import get_peft_model, LoraConfig, TaskType

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
# 注意: 保留这些自定义的 modeling 文件
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM


class HLLM(BaseModel):
    """
    HLLM 模型，已适配为仅使用 UserLLM 和预计算物品嵌入的模式。
    [QLoRA 最终修复版]
    """
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM, self).__init__()
        self.logger = getLogger()
        self.config = config # 保存config以备后用

        self.user_pretrain_dir = config['user_pretrain_dir']
        
        # --- [核心修改] 初始化 UserLLM ---
        self.logger.info(f"正在从以下路径创建 User LLM: {self.user_pretrain_dir}")
        
        if config.get('use_qlora', False):
            self.logger.info("检测到 use_qlora: true，正在使用 QLoRA 模式加载 UserLLM...")
            
            # 1. 配置4-bit量化 (使用 float16 以兼容您的GPU)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # 2. 以4-bit量化方式加载基础模型
            #    我们复用 create_llm 的逻辑来获取正确的模型类
            model_class = self._get_model_class(self.user_pretrain_dir)
            base_model = model_class.from_pretrained(
                self.user_pretrain_dir,
                quantization_config=quantization_config,
                device_map="auto",
            )
            
            # 3. 配置LoRA参数
            lora_config = LoraConfig(
                r=config['qlora_r'],
                lora_alpha=config['qlora_alpha'],
                lora_dropout=config['qlora_dropout'],
                target_modules=config['qlora_target_modules'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # 4. 将LoRA适配器应用到量化后的模型上
            self.user_llm = get_peft_model(base_model, lora_config)
            
            self.logger.info("QLoRA UserLLM 加载完成。可训练参数如下：")
            self.user_llm.print_trainable_parameters()
            
        else:
            self.logger.info("正在使用全参数模式加载 UserLLM...")
            self.user_llm = self.create_llm(self.user_pretrain_dir, init=config.get('user_llm_init', True))

        
        
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.logger.info("模型已配置 Prediction Head 和 BCEWithLogitsLoss。")

    def _get_model_class(self, pretrain_dir):
        """辅助函数，根据配置获取正确的模型类"""
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        model_map = {
            "llama": LlamaForCausalLM, "mistral": MistralForCausalLM,
            "bert": BertModel, "baichuan": BaichuanForCausalLM
        }
        model_type = getattr(hf_config, "model_type", "")
        return model_map.get(model_type, AutoModelForCausalLM)

    def create_llm(self, pretrain_dir, init=True):
        """用于实例化大语言模型的辅助函数（非QLoRA模式）"""
        model_class = self._get_model_class(pretrain_dir)
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        hf_config.gradient_checkpointing = self.config.get('gradient_checkpointing', False)
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True
        hf_config.use_ft_flash_attn = self.config.get('use_ft_flash_attn', False)
        
        if init:
            return model_class.from_pretrained(pretrain_dir, config=hf_config)
        else:
            return model_class(config=hf_config)

    def _get_user_representation(self, history_embeds, attention_mask):
        """辅助函数，用于从历史记录中提取用户表征"""
        outputs = self.user_llm(
            inputs_embeds=history_embeds, 
            attention_mask=attention_mask,
            output_hidden_states=True, 
        )
        # 兼容不同模型的输出格式
        last_hidden_state = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        
        last_token_indices = attention_mask.sum(dim=1) - 1
        user_representation = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_token_indices]
        return user_representation

    def forward(self, interaction):
        """
        [最终版] 用于训练的前向传播函数 (使用点积)。
        """
        history_embeds = interaction['history_item_embeddings']
        attention_mask = interaction['attention_mask']
        candidate_embeds = interaction['candidate_item_embeddings']
        labels = interaction['labels']

        # 1. 提取用户表征
        user_representation = self._get_user_representation(history_embeds, attention_mask)

        # 2. [修改] 使用点积计算所有候选物品的分数
        #    扩展用户表征以匹配候选物品的数量: (batch, 1, dim)
        user_rep_expanded = user_representation.unsqueeze(1)
        
        #    计算点积: (batch, 1, dim) * (batch, num_candidates, dim) -> (batch, num_candidates, dim)
        #    然后沿维度-1求和，得到最终分数 (batch, num_candidates)
        scores = (user_rep_expanded * candidate_embeds).sum(dim=-1)

        # 3. 计算损失
        loss = self.loss_fct(scores, labels.float())
        
        return {'loss': loss}

    @torch.no_grad()
    def predict(self, interaction):
        """
        [最终版] 用于预测/评估的前向传播函数 (使用点积)。
        """
        history_embeds = interaction['history_item_embeddings']
        attention_mask = interaction['attention_mask']
        candidate_embeds = interaction['candidate_item_embeddings']
        
        # 1. 提取用户表征
        user_representation = self._get_user_representation(history_embeds, attention_mask)

        # 2. [修改] 使用点积计算所有候选物品的分数
        user_rep_expanded = user_representation.unsqueeze(1)
        scores = (user_rep_expanded * candidate_embeds).sum(dim=-1)
        
        return scores