# [HSFT: HLLM SIMPLE FINE TUNING FOR USERLLM]

## Introduction
本项目将原始的HLLM的UserLLM部分进行微调。在公开数据集Pixel200K上，对一个已针对推荐任务预训练的UserLLM（TinyLlama-1.1B）进行二次微调，验证其可行性与效果。将原始代码进行更改适配，并能够在消费级GPU上进行微调。


## Installation
1. 安装 `pip3 install -r requirements.txt`. 
```
pytorch==2.1.0
deepspeed==0.14.2
transformers==4.41.1
lightning==2.4.0
flash-attn==2.5.9post1
fbgemm-gpu==0.5.0 [optional for HSTU]           (在本项目不需要)
sentencepiece==0.2.0 [optional for Baichuan2]   (在本项目不需要)
```
2. 下载 `PixelRec` 数据集:
    1. 下载 `PixelRec` [PixelRec](https://github.com/westlake-repl/PixelRec)并放至对应目录, 只需要一个Pixel200K即可.
    2. 目录结构:
        ```bash
        ├── dataset # Store Interactions
        │   ├── amazon_books.csv
        │   ├── Pixel1M.csv
        │   ├── Pixel200K.csv
        │   └── Pixel8M.csv
        └── information # Store Item Information
            ├── Pixel1M.csv
            ├── Pixel200K.csv
            └── Pixel8M.csv
        ``` 
3. 下载 `User LLM`和`Item LLM`的预训练模型:
直接运行download_itemllm.py和download_userllm.py即可
    

## Item Caching
将Item先通过预训练的Item LLM生成对应的ITEM Embedding作为缓存。我们只进行UserLLM的微调，所以这里我们直接将生成的缓存作为数据集。
直接运行item_caching.py即可。

## DataSet

1. 这里已经将 `code/REC/data/dataset/trainset.py`和`code/REC/data/dataset/testset.py`新增了`CachedEmbeddingTrainDataset`和`CachedEmbeddingEvalDataset` 用于处理预计算好的Embedding缓存。
2. 同时更改collate_fn.py能够适配训练集/测试集，同时适配负采样。

## Config and BUG
1. 配置都在`code/HLLM`和`code/overall`下的yaml文件中，可以自定义超参数
2. 根据硬件选择是否使用flash-attn以及一些QLora精度的选择

## Eval
可以直接测试Base UserLLM的效果
    
```python
python code/main.py --config_file code/overall/LLM_ddp.yaml code/HLLM/HLLM.yaml --dataset Pixel200K --loss bce --use_item_cache True --item_cache_path dataset/pixel200k_item_embeddings.pt --user_pretrain_dir model_checkpoints/UserLLM_complete/ --item_pretrain_dir model_checkpoints/ItemLLM_complete/ --val_only True
```

## Fine-tuning
在训练集上使用QLora微调
```python
python code/main.py  --config_file code/overall/LLM_ddp.yaml code/HLLM/HLLM.yaml  --dataset Pixel200K  --loss bce  --use_item_cache True  --item_cache_path dataset/pixel200k_item_embeddings.pt  --user_pretrain_dir model_checkpoints/UserLLM_complete/ --use_qlora True
```

# Result


| (Metric)      | Baseline | fine-tune |            | 
| ------------- | -------- |---------- | ---------- | 
| Recall@5      | 0.078000 | 0.078995  | +1.28%     | 
| Recall@10     | 0.144515 | 0.144615  | +0.07%     | 
| Recall@20     | 0.265385 | 0.272355  | +2.63%     | 
| Recall@AVG    |    -     |    -      |            | 
| NDCG@5        | 0.047612 | 0.048935  | +2.78%     |
| NDCG@10       | 0.068872 | 0.069930  | +1.54%     |
| NDCG@20       | 0.099094 | 0.101837  | +2.77%     |
| NDCG@AVG      |          |           |            |

