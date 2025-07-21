# download_userllm.py
# 这是一个专门用于下载、整合并保存UserLLM的独立脚本。
# [V2 - 空间优化版] 此版本经过优化，可在有限的磁盘空间（如12GB）下运行。

import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from collections import OrderedDict

# --- 配置参数 ---
# 1. HLLM微调权重的仓库地址
FINETUNED_REPO_ID = "ByteDance/HLLM"
# 我们要使用的权重文件名
FINETUNED_FILENAME = "1B_Pixel8M/pytorch_model.bin"

# 2. HLLM所基于的基础模型地址
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# 3. 定义最终要保存UserLLM的本地路径
SAVE_DIRECTORY = "../model_checkpoints/UserLLM_complete"

def clear_hf_cache():
    """一个辅助函数，用于清理Hugging Face的下载缓存。"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        print(f"清理Hugging Face缓存目录: {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("缓存清理完成！")

def main():
    """主执行函数"""
    print("开始下载、整合并保存 UserLLM (空间优化模式)...")
    
    # 确保目标文件夹存在
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    
    # --- [关键优化] 在开始前，先清理一次缓存，确保有最大可用空间 ---
    clear_hf_cache()

    # --- 第一步: 只下载并加载基础模型的配置文件和分词器 ---
    print(f"\n1. 正在从 '{BASE_MODEL_ID}' 加载基础模型架构和分词器...")
    # from_pretrained 会自动下载并缓存所需文件
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    # 我们只加载一个空的模型结构(在内存中)，不下载4.4G的权重
    model = AutoModel.from_config(config)
    print("基础模型架构和分词器加载完成。")

    # --- 第二步: 下载HLLM的微调权重 ---
    print(f"\n2. 正在从 '{FINETUNED_REPO_ID}' 下载HLLM的共享微调权重文件 (约8.8GB)...")
    finetuned_weights_path = hf_hub_download(
        repo_id=FINETUNED_REPO_ID,
        filename=FINETUNED_FILENAME
    )
    print("微调权重下载完成。")

    # --- 第三步: 将权重加载到内存，并立刻清理缓存 ---
    print("\n3. 正在将权重加载到内存中...")
    hllm_state_dict = torch.load(finetuned_weights_path, map_location="cpu")
    print("权重已加载到内存。")
    
    # --- [关键优化] 立刻清理缓存，释放8.8GB空间 ---
    clear_hf_cache()

    # --- 第四步: 在内存中处理权重密钥，只提取User LLM的部分 ---
    print("\n4. 正在处理权重，提取User LLM部分并重命名密钥...")
    user_llm_state_dict = OrderedDict()
    prefix = "user_llm.model."
    
    for key, value in hllm_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            user_llm_state_dict[new_key] = value
    
    if not user_llm_state_dict:
        raise ValueError(f"没有找到任何以 '{prefix}' 为前缀的权重，请检查模型文件！")

    print("密钥处理完成！")

    # --- 第五步: 将处理好的权重加载到基础模型中 ---
    print("\n5. 正在将处理后的User LLM权重加载到模型上...")
    model.load_state_dict(user_llm_state_dict)
    print("权重加载成功！")

    # --- 第六步: 将整合后的完整模型和分词器保存到本地 ---
    # 此时硬盘空间充足，可以安全地写入约3.9GB的最终模型
    print(f"\n6. 正在将完整的User LLM模型和分词器保存到 '{SAVE_DIRECTORY}'...")
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    model.save_pretrained(SAVE_DIRECTORY)
    print("所有操作完成！您现在拥有一个完整的、自给自足的本地 User LLM模型副本。")

if __name__ == "__main__":
    main()
