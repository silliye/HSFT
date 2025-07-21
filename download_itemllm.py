import os
import torch
import shutil
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from collections import OrderedDict

# --- 配置参数 ---
FINETUNED_REPO_ID = "ByteDance/HLLM"
FINETUNED_FILENAME = "1B_Pixel8M/pytorch_model.bin"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SAVE_DIRECTORY = "../model_checkpoints/HLLM-1B-Pixel8M_complete"

def main():
    print("开始下载、整合并保存模型...")
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    # --- 第一步: 加载基础模型架构和分词器 ---
    print(f"\n1. 正在从 '{BASE_MODEL_ID}' 加载基础模型架构和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    model = AutoModel.from_config(config) # 只加载空壳结构
    print("基础模型架构和分词器加载完成。")

    # --- 第二步: 下载HLLM的微调权重 ---
    print(f"\n2. 正在从 '{FINETUNED_REPO_ID}' 下载HLLM的微调权重文件...")
    finetuned_weights_path = hf_hub_download(
        repo_id=FINETUNED_REPO_ID,
        filename=FINETUNED_FILENAME
    )
    print("微调权重下载完成。")

    # --- 第三步: 处理权重密钥，只提取Item LLM的部分 ---
    print("\n3. 正在处理权重，提取Item LLM部分并重命名密钥...")
    hllm_state_dict = torch.load(finetuned_weights_path, map_location="cpu")
    item_llm_state_dict = OrderedDict()
    prefix = "item_llm.model."
    for key, value in hllm_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            item_llm_state_dict[new_key] = value
    print("密钥处理完成！")

    # --- 第四步: 将处理好的权重加载到基础模型中 ---
    print("\n4. 正在将处理后的Item LLM权重加载到模型上...")
    model.load_state_dict(item_llm_state_dict)
    print("权重加载成功！")

    # --- 第五步: 清理Hugging Face缓存，释放磁盘空间 ---
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        print(f"\n5. 清理Hugging Face缓存目录: {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("缓存清理完成！")

    # --- 第六步: 将整合后的完整模型和分词器保存到本地 ---
    print(f"\n6. 正在将完整的Item LLM模型和分词器保存到 '{SAVE_DIRECTORY}'...")
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    model.save_pretrained(SAVE_DIRECTORY)
    print("所有操作完成！")

if __name__ == "__main__":
    main()