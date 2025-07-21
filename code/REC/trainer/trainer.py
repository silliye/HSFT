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
import sys
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import deepspeed

# [QLoRA 修复] 导入 bitsandbytes 库
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from REC.data.dataset import BatchTextDataset
from REC.data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from REC.evaluator import Evaluator, Collector
from REC.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    get_tensorboard, set_color, get_gpu_usage, WandbLogger
from REC.utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy


class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.skippingCount = 0
        self.wandblogger = WandbLogger(config)

        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']

        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0 # 在单卡模式下，默认rank为0

        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)

        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)
        
        if self.config.get('use_qlora', False):
            # 在QLoRA模式下，"best model file" 指的是最佳适配器保存的目录
            self.saved_model_file = os.path.join(self.checkpoint_dir, "best_adapter")
        else:
            self.saved_model_name = '{}-{}.pth'.format(self.config['model'], 0)
            self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)
            
        

        self.use_text = config['use_text']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.scheduler_config = config['scheduler_args']
        if config['freeze_prefix'] or config['freeze_ad']:
            freeze_prefix = config['freeze_prefix'] if config['freeze_prefix'] else []
            if config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config['ft_item']:
                freeze_prefix.extend(['item_embedding'])

            self._freeze_params(freeze_prefix)

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config['type'] == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        """
        [QLoRA 最终修复]
        根据是否使用QLoRA，选择不同的优化器。
        并集成了原始代码中复杂的参数分组和学习率设置逻辑。
        """
        if self.config.get('use_qlora', False):
            self.logger.info("Using QLoRA-compatible PagedAdamW8bit optimizer.")
            if bnb is None:
                raise ImportError("bitsandbytes is required for QLoRA. Please install it with `pip install bitsandbytes`")
            # For QLoRA, we use a specific optimizer from bitsandbytes.
            # PEFT ensures that model.parameters() only returns the trainable adapter parameters.
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            return bnb.optim.PagedAdamW8bit(
                list(params),
                lr=self.optim_args['learning_rate'],
                weight_decay=self.optim_args['weight_decay']
            )
        else:
            # If not using QLoRA, use the original, more complex optimizer setup.
            self.logger.info("Using standard optimizer setup.")
            if len(self.optim_args) == 4:
                params = self.model.named_parameters()
                modal_params = []
                recsys_params = []
                modal_decay_params = []
                recsys_decay_params = []
                decay_check_name = self.config['decay_check_name']
                for index, (name, param) in enumerate(params):
                    if param.requires_grad:
                        if 'visual_encoder' in name:
                            modal_params.append(param)
                        else:
                            recsys_params.append(param)
                        if decay_check_name:
                            if decay_check_name in name:
                                modal_decay_params.append(param)
                            else:
                                recsys_decay_params.append(param)
                if decay_check_name:
                    optimizer = optim.AdamW([
                        {'params': modal_decay_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                        {'params': recsys_decay_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                    ])
                else:
                    optimizer = optim.AdamW([
                        {'params': modal_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                        {'params': recsys_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                    ])
            elif self.config.get('lr_mult_prefix') and self.config.get('lr_mult_rate'):
                normal_params_dict = {
                    "params": [],
                    "lr": self.optim_args['learning_rate'],
                    "weight_decay": self.optim_args['weight_decay']
                }
                high_lr_params_dict = {
                    "params": [],
                    "lr": self.optim_args['learning_rate'] * self.config['lr_mult_rate'],
                    "weight_decay": self.optim_args['weight_decay']
                }
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        if any(n.startswith(x) for x in self.config['lr_mult_prefix']):
                            high_lr_params_dict["params"].append(p)
                        else:
                            normal_params_dict["params"].append(p)
                optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
            elif self.config.get('optimizer_kwargs'):
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                self.config['optimizer_kwargs']['optimizer']['params']['lr'] = self.optim_args['learning_rate']
                self.config['optimizer_kwargs']['optimizer']['params']['weight_decay'] = self.optim_args['weight_decay']
                optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, **self.config['optimizer_kwargs']['optimizer']['params'])
            else:
                params = filter(lambda p: p.requires_grad, self.model.parameters())
                optimizer = optim.AdamW(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            return optimizer

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        train_data = train_data
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(
                total=len(train_data),
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'),
                file=sys.stdout
            )
        bwd_time = t.time()
        for batch_idx, data in enumerate(train_data):
            start_time = bwd_time
            self.optimizer.zero_grad()
            data = self.to_device(data)
            data_time = t.time()
            losses_dict = self.model(data)
            fwd_time = t.time()
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses_dict = model_out.pop('loss')
            losses = losses_dict['loss']
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            self.lite.backward(losses)
            grad_norm = self.optimizer.step()
            bwd_time = t.time()
            if self.scheduler_config:
                self.lr_scheduler.step()
            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f} bwd: {bwd_time-fwd_time:.3f}"
                if self.scheduler_config:
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                if self.config['loss'] and self.config['loss'] == 'nce':
                    for k, v in model_out.items():
                        msg += f" {k}: {v:.3f}"
                if grad_norm:
                    msg = msg + f" grad_norm: {grad_norm.sum():.4f}"
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)
                self.logger.info("\n" + "-"*50)
            if self.config['debug'] and batch_idx >= 10:
                break

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        if self.config.get('use_qlora', False):
            # QLoRA模式：只保存轻量级的适配器权重
            if self.config.get('use_qlora', False):
            # ‼️ 【最终修复】: 总是保存到 self.saved_model_file 定义的固定路径
                adapter_save_path = self.saved_model_file
                peft_model_to_save = self.model.module.user_llm
                peft_model_to_save.save_pretrained(adapter_save_path)
                if self.rank == 0 and verbose:
                    self.logger.info(set_color('Saving best QLoRA adapter to', 'blue') + f': {adapter_save_path}')
        else:
            # 原始模式：保存完整的模型和优化器状态
            state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }
            self.lite.save(os.path.join(self.checkpoint_dir, self.saved_model_name), state=state)
            if dist.get_rank() == 0 and verbose:
                self.logger.info(set_color('Saving full checkpoint to', 'blue') + f': {self.saved_model_file}')


    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@', '_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def to_device(self, data):
        device = self.device
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        if isinstance(data, tuple):
            return tuple(self.to_device(v) for v in data)
        if isinstance(data, list):
            return [self.to_device(v) for v in data]
        # 对于其他类型（如int, float, str），直接返回
        return data

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        
        if self.scheduler_config:
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            tot_steps = len(train_data) * self.epochs
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
        nnodes = world_size // local_world_size
        precision = self.config['precision'] if self.config['precision'] else '32'
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            strategy = DeepSpeedStrategy(stage=self.config["stage"], precision=precision)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        else:
            self.logger.info(f"Use DDP strategy")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        self.lite.launch()
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if self.config['auto_resume']:
            raise NotImplementedError
        self.model.mark_forward_method('predict')
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            
            # train
            if self.config['need_training'] == None or self.config['need_training']:
                 # ‼️ 【最终修复】增加一个判断，只在分布式采样器存在时才调用 set_epoch
                if hasattr(train_data.sampler, 'set_epoch'):
                    train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                

                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@', '_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                        (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def _cached_embedding_batch_eval(self, batched_data):
        """
        专门为处理缓存Embedding数据格式而设计的新评测函数。
        """
        # 1. 将数据字典移动到正确的设备
        interaction = self.to_device(batched_data)
        history_tensor = interaction['history_item_embeddings']
        # ‼️ 【核心修复】: 在调用模型前，确保输入数据的类型与模型一致
        #    我们直接从模型参数上获取它期望的 dtype
        expected_dtype = next(self.model.parameters()).dtype

        # 检查并转换 interaction 字典中所有 embedding 张量的数据类型
        interaction['history_item_embeddings'] = interaction['history_item_embeddings'].to(expected_dtype)
        interaction['candidate_item_embeddings'] = interaction['candidate_item_embeddings'].to(expected_dtype)
        
        # 2. 调用我们之前修改好的 HLLM 模型的 predict 方法
        #    这个 predict 方法接收的就是一个包含 embeddings 的字典
        scores = self.model.predict(interaction)
        
        # 3. 从数据中获取真实目标物品的索引
        #    根据 CachedEmbeddingEvalDataset 的设计，正样本始终在第0位
        positive_i = interaction['target_index']
        
        # 4. 创建一个从0到batch_size-1的张量，代表每个用户的索引
        positive_u = torch.arange(scores.shape[0], device=self.device)

        # 5. 返回模型分数和真实目标，格式与旧函数保持一致，以便下游Collector处理
        return scores, positive_u, positive_i
    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        user, time_seq, history_index, positive_u, positive_i = batched_data 
        interaction = self.to_device(user)
        time_seq = self.to_device(time_seq)
        if self.config['model'] == 'HLLM':
            if self.config['stage'] == 3:
                scores = self.model.module.predict(interaction, time_seq, self.item_feature)
            else:
                scores = self.model((interaction, time_seq, self.item_feature), mode='predict')
        else:
            scores = self.model.module.predict(interaction, time_seq, self.item_feature)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        # --- 修改开始 ---
        # 1. 检查配置文件中是否有缓存路径
        embedding_cache_path = self.config.get('item_cache_path', None)


        # 2. 如果有缓存路径，就直接加载，跳过所有计算
        if embedding_cache_path and os.path.exists(embedding_cache_path):
            self.logger.info(f"Loading item features directly from cache for evaluation: {embedding_cache_path}")
            
            # 1. 正常加载您的字典缓存
            self.item_feature = torch.load(embedding_cache_path, map_location=self.device)
            # ‼️ 在这里打印加载后的第一个 embedding 的数据类型
            first_value = next(iter(self.item_feature.values()))
            self.logger.info(f"DEBUG (Trainer): 缓存加载完成。第一个Embedding的Dtype是: {first_value.dtype}")
            # 2. 从模型配置中获取嵌入的维度 (最稳妥的方式)
            hidden_size = self.model.user_llm.config.hidden_size
            
            # 3. 手动添加或覆盖 padding item (键为 0)
            # 无论原始缓存中是否有键 0，我们都确保它现在的值是一个全零向量
            self.item_feature[0] = torch.zeros(hidden_size, device=self.device)
            
            # 4. 记录日志并返回
            self.logger.info("Item features for evaluation loaded successfully, padding item manually handled.")
            return

        # 3. 如果没有缓存路径，才执行原来的实时计算逻辑
        self.logger.info("No item_embedding_path found. Computing item features on-the-fly...")
        if self.use_text:
            item_data = BatchTextDataset(config, data)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=14, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')
                    self.item_feature.append(items)
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                if self.config['stage'] == 3:
                    self.item_feature = self.item_feature.bfloat16()
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()
        # --- 修改结束 ---

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False):
        if not eval_data:
            return
        if init_model:
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
            nnodes = world_size // local_world_size
            if self.config['strategy'] == 'deepspeed':
                self.logger.info(f"Use deepspeed strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DeepSpeedStrategy(stage=self.config['stage'], precision=precision)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
            else:
                self.logger.info(f"Use DDP strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DDPStrategy(find_unused_parameters=True)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model = self.lite.setup(self.model)
                
                self.model.mark_forward_method('predict')

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            if self.config.get('use_qlora', False):
                self.logger.info(f"Loading best QLoRA adapter from {checkpoint_file}")
                if not os.path.isdir(checkpoint_file):
                    raise FileNotFoundError(f"QLoRA adapter directory not found: {checkpoint_file}")
                # 使用 .module 访问被 Fabric 包装的原始 PEFT 模型
                peft_model_to_load = self.model.module.user_llm
                peft_model_to_load.load_adapter(checkpoint_file, "default")
                message_output = f'Successfully loaded QLoRA adapter from {checkpoint_file}'
            else:
                if not os.path.exists(checkpoint_file): 
                    raise FileNotFoundError(f"Saved model file not found: {checkpoint_file}")
                state = {"model": self.model}
                self.lite.load(checkpoint_file, state)
                message_output = f'Loading model from {checkpoint_file}'
            self.logger.info(message_output)
        self.logger.info(f"DEBUG: 转换前，模型参数的Dtype是: {next(self.model.parameters()).dtype}")
        
        precision = self.config.get('precision', '32')
        if precision in ['16', '16-mixed']:
            self.logger.info("检测到 '16'/'16-mixed' 精度，正在将模型转换为 float16 (.half())...")
            self.model = self.model.half()
        elif precision in ['bf16', 'bf16-mixed']:
            self.logger.info("检测到 'bf16'/'bf16-mixed' 精度，正在将模型转换为 bfloat16 (.bfloat16())...")
            self.model = self.model.bfloat16()

        self.logger.info(f"DEBUG: 转换后，模型参数的Dtype是: {next(self.model.parameters()).dtype}")
        # ‼️‼️ 修复代码结束
        with torch.no_grad():
            self.model.eval()
            if self.config.get('use_item_cache', False):
                # 如果配置中启用了 use_item_cache，就使用我们为缓存模式写的新函数
                eval_func = self._cached_embedding_batch_eval
                self.logger.info("Using cached embedding evaluation function.")
            else:
                # 否则，继续使用旧的基于ID的评测函数
                eval_func = self._full_sort_batch_eval
                self.logger.info("Using full sort (ID-based) evaluation function.")

            self.tot_item_num = eval_data.dataset.dataload.item_num
            self.compute_item_feature(self.config, eval_data.dataset.dataload)
            


            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", 'pink'),
                    file=sys.stdout
                ) if show_progress and self.rank == 0 else eval_data
            )
            fwd_time = t.time()
            for batch_idx, batched_data in enumerate(iter_data):
                if batched_data is None:
                    # self.logger.warning(f"Skipping batch {batch_idx} due to all samples being invalid.")
                    self.skippingCount += 1
                    continue
                start_time = fwd_time
                data_time = t.time()
                scores, positive_u, positive_i = eval_func(batched_data)
                fwd_time = t.time()

                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(f"data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}", refresh=False)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)
            print('跳过了', self.skippingCount)

            num_total_examples = len(eval_data.dataset)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)

            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')
            print('跳过了', self.skippingCount)
            return result
