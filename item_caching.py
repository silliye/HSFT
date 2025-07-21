# cache_items.py (终极版)
# 该脚本负责加载ItemLLM，添加特殊词元，永久升级模型，
# 并为所有物品生成Embedding缓存，为后续实验做好准备。

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# --- 配置参数 ---
# MODEL_PATH: 我们在第一阶段整合好的、完整的本地ItemLLM模型路径
MODEL_PATH = "../model_checkpoints/ItemLLM_complete/" 

# ITEM_INFO_PATH: 物品信息CSV文件路径
ITEM_INFO_PATH = "../information/Pixel200K.csv" 

# CACHE_SAVE_PATH: 最终生成的Embedding缓存文件的保存路径
CACHE_SAVE_PATH = "../dataset/pixel200k_item_embeddings222.pt" 

# DEVICE: 自动检测使用GPU还是CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BATCH_SIZE: 批处理大小，可以根据您的GPU显存进行调整 (例如16, 32, 64)
BATCH_SIZE = 32 

def main():
    """主执行函数"""
    print(f"使用的设备: {DEVICE}")
    print(f"正在从本地路径加载Item LLM模型和分词器: {MODEL_PATH}")

    # 1. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    item_llm = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)

    # 2. [终极优化] 添加特殊词元并重设词表大小，实现永久升级
    print("正在添加[ITEM]特殊词元并更新模型...")
    special_tokens_dict = {'additional_special_tokens': ['[ITEM]']}
    # add_special_tokens会返回新添加的词元数量
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    if num_added_toks > 0:
        # 如果成功添加了新词元，就需要重设模型Embedding层的大小以匹配
        item_llm.resize_token_embeddings(len(tokenizer))
        print(f"词表大小已从 {len(tokenizer) - num_added_toks} 调整为 {len(tokenizer)}")
        
        # [关键] 将修改后的模型和分词器保存回去，实现永久化
        print(f"正在将更新后的模型和分词器保存回: {MODEL_PATH}")
        tokenizer.save_pretrained(MODEL_PATH)
        item_llm.save_pretrained(MODEL_PATH)
        print("模型和分词器已更新并永久保存！")
    else:
        print("特殊词元已存在，无需更新。")

    # 3. 检查并设置padding token，解决padding报错问题
    if tokenizer.pad_token is None:
        print("未找到padding token, 正在将其设置为eos_token...")
        tokenizer.pad_token = tokenizer.eos_token
        # 同样保存一下，确保设置被持久化
        tokenizer.save_pretrained(MODEL_PATH)

    # 4. 将模型设置为评估模式
    item_llm.eval()

    # 5. 读取并准备物品数据
    print(f"正在读取物品信息从: {ITEM_INFO_PATH}")
    df_items = pd.read_csv(ITEM_INFO_PATH).drop_duplicates(subset=['item_id'])
    df_items['text'] = df_items['title'].fillna('') + ' ' + df_items['tag'].fillna('') + ' ' + df_items['description'].fillna('')
    
    items = df_items['item_id'].tolist()
    texts = df_items['text'].tolist()
    
    print(f"共找到 {len(items)} 个独立物品需要处理。")

    # 6. 批量生成并保存Embedding
    all_embeddings = {}
    # 使用torch.no_grad()可以关闭梯度计算，节省显存并加速
    with torch.no_grad():
        # 使用tqdm创建进度条
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="正在生成Embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_items = items[i:i+BATCH_SIZE]
            
            # 在文本末尾添加我们新定义的特殊token
            texts_with_special_token = [text + " [ITEM]" for text in batch_texts]

            # 使用分词器处理文本，并发送到GPU
            inputs = tokenizer(texts_with_special_token, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            
            # 模型前向传播
            outputs = item_llm(**inputs, output_hidden_states=True)
            
            # 精确提取最后一个有效token的Embedding
            last_hidden_state = outputs.hidden_states[-1]
            # 通过attention_mask找到每个序列的真实长度，减1得到最后一个token的索引
            sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
            # 使用高级索引精确提取
            embeddings = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths].cpu()
            
            # 将结果存入字典
            for item_id, emb in zip(batch_items, embeddings):
                all_embeddings[item_id] = emb

    # 7. 保存最终的Embedding缓存文件
    print(f"正在保存缓存文件到: {CACHE_SAVE_PATH}")
    torch.save(all_embeddings, CACHE_SAVE_PATH)
    print("缓存文件保存成功！第二阶段完成。")

if __name__ == "__main__":
    main()
