import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

try :
    from torch.cuda.amp import autocast, GradScaler
    #from torch.amp import autocas
    load_amp = True
except:
    load_amp = False


class _PSNR(torch.nn.Module):
    def forward(self, pred, gt, eps=1e-12):
        mse = torch.mean((pred - gt) ** 2)
        return 100.0 if mse.item() < eps else (-10.0 * torch.log10(mse)).item()

class EglliePSNR(torch.nn.Module):
    def __init__(self): super().__init__(); self.psnr=_PSNR()
    def forward(self, pred,gt): return self.psnr(pred, gt)

def visualize_event_image(voxel_grid, height, width,savepath=None,saveimg=False,show=False):
    """功能1：生成累积的事件图像（空间投影图）"""
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    #assert voxel_grid.ndim == 3

    full_frame = np.zeros((height, width), dtype=np.float32)
    for t_bin in range(voxel_grid.shape[0]):
        full_frame += voxel_grid[t_bin]

    if saveimg:

        plt.imsave(savepath, full_frame, cmap='seismic', vmin=-2, vmax=2)
    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(full_frame, cmap='seismic', vmin=-2, vmax=2)
        plt.axis('off')  # 不显示坐标轴或边框
        plt.title('1')
        plt.show()

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel_Bilevel_fusion(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel_Bilevel_fusion, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
                  
        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.m_psnr = EglliePSNR().cuda()
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define Rnet losses
        if train_opt.get('pixel_opt_list'):
            self.cri_pix_list = []
            self.cri_pix_weight = []
            self.cri_pix_pred_key = []
            self.cri_pix_target_key = []

            for loss_cfg in train_opt['pixel_opt_list']:
                loss_type = loss_cfg.pop('type')
                weight = loss_cfg.pop('weight', 1.0)
                pred_key = loss_cfg.pop('pred_key')
                target_key = loss_cfg.pop('target_key')

                # 获取对应的loss类
                loss_cls = getattr(loss_module, loss_type)

                # 实例化 loss
                self.cri_pix_list.append(loss_cls(**loss_cfg).to(self.device))
                self.cri_pix_weight.append(weight)
                self.cri_pix_pred_key.append(pred_key)
                self.cri_pix_target_key.append(target_key)
        else:
            raise ValueError('pixel loss are None.')

        # define Evnet losses

        if train_opt.get('voxel_loss_list'):

            self.cri_voxel_list = []
            self.cri_voxel_weight = []
            self.cri_voxel_pred_key = []
            self.cri_voxel_target_key = []
            for loss_cfg in train_opt['voxel_loss_list']:
                loss_type = loss_cfg.pop('type')
                weight = loss_cfg.pop('weight', 1.0)
                pred_key = loss_cfg.pop('pred_key')
                target_key = loss_cfg.pop('target_key')
                loss_cls = getattr(loss_module, loss_type)
                self.cri_voxel_list.append(loss_cls(**loss_cfg).to(self.device))
                self.cri_voxel_weight.append(weight)
                self.cri_voxel_pred_key.append(pred_key)
                self.cri_voxel_target_key.append(target_key)
        else:
            raise ValueError('voxel loss list is None.')
        # set up optimizers and schedulers

        if train_opt.get('r_related_pixel_opt_list'):

            self.cri_r_related_list = []
            self.cri_r_related_weight = []
            self.cri_r_related_pred_key = []
            self.cri_r_related_target_key = []
            for loss_cfg in train_opt['r_related_pixel_opt_list']:
                loss_type = loss_cfg.pop('type')
                weight = loss_cfg.pop('weight', 1.0)
                pred_key = loss_cfg.pop('pred_key')
                target_key = loss_cfg.pop('target_key')
                loss_cls = getattr(loss_module, loss_type)
                self.cri_r_related_list.append(loss_cls(**loss_cfg).to(self.device))
                self.cri_r_related_weight.append(weight)
                self.cri_r_related_pred_key.append(pred_key)
                self.cri_r_related_target_key.append(target_key)
        else:
            raise ValueError('voxel loss list is None.')
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        params_lr = []  # L & R 分支
        params_e = []  # E 分支

        for name, p in self.net_g.named_parameters():
            if not p.requires_grad:
                continue

            # L & R params (Lnet和Rnet的所有参数)
            if 'Lnet' in name or 'Rnet' in name:
                params_lr.append(p)

            # E params (Enet的所有参数)
            elif 'Enet' in name:
                params_e.append(p)

            else:
                logger = get_root_logger()
                logger.warning(f'Params {name} will not be optimized.')

        opt_cfg_i = train_opt['optim_i'].pop('type')
        opt_cfg_e = train_opt['optim_e'].pop('type')

        if opt_cfg_i == 'Adam':
            self.optimizer_i = torch.optim.Adam(
                params_lr, **train_opt['optim_i'])
        elif opt_cfg_i == 'AdamW':
            self.optimizer_i = torch.optim.AdamW(
                params_lr, **train_opt['optim_i'])
        else:
            raise NotImplementedError(
                f'optimizer {opt_cfg_i} is not supperted yet.')

        if opt_cfg_e == 'Adam':
            self.optimizer_e = torch.optim.Adam(
                params_e, **train_opt['optim_e'])
        elif opt_cfg_e == 'AdamW':
            self.optimizer_e = torch.optim.AdamW(
                params_e, **train_opt['optim_e'])
        else:
            raise NotImplementedError(
                f'optimizer {opt_cfg_e} is not supperted yet.')

        self.optimizers.append(self.optimizer_i)
        self.optimizers.append(self.optimizer_e)

        #self.is_update_img = True   #确认优化LLIE还是优化Event
    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.event = data['event'].to(self.device)
        self.event_denoising = data['event_denoising'].to(self.device)
        self.mask = data['event_denoising'].to(self.device)  #
        self.mask[self.mask > 0] = 1  # pos
        self.mask[self.mask < 0] = 2  # neg
        self.mask = self.mask.long()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.event = data['event'].to(self.device)
       #self.event_denoising = data['event_denoising'].to(self.device)
        #self.mask = data['event_denoising'].to(self.device)  #
        # self.mask[self.mask > 0] = 1  # pos
        # self.mask[self.mask < 0] = 2  # neg
        # self.mask = self.mask.long()


        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = data['lq'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        # ==================== 阶段1: 下层优化（图像分支 L & R）====================
        self.optimizer_i.zero_grad()

        # 🔥 1. 分别保存L和R的参数
        l_params = []  # L分支的参数
        r_params = []  # R分支的参数
        l_params_old = []
        r_params_old = []

        # 【1】保存下层的原始参数
        for name, p in self.net_g.named_parameters():
            # L分支参数
            if 'Lnet' in name:
                l_params.append(p)
                l_params_old.append(p.data.clone())
            # R分支参数（包括Rnet中所有参数）
            elif 'Rnet' in name:
                r_params.append(p)
                r_params_old.append(p.data.clone())

        # 下层参数 = L + R
        lower_params = l_params + r_params

        # 【2】下层进行优化（R和L联合更新）
        with autocast('cuda', enabled=self.use_amp):
            self.net_g.mode = 'train'
            preds, I, R, event = self.net_g(self.lq, self.event, self.gt)

            outputs = {
                "preds": preds[-1] if isinstance(preds, list) else preds,
                "I": I,
                "R": R,
                "gt": self.gt,
                "pred_voxel": event
            }

            # 计算下层损失（图像相关）
            loss_dict = OrderedDict()
            lower_loss = 0.0

            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_pix_list, self.cri_pix_weight,
                    self.cri_pix_pred_key, self.cri_pix_target_key):

                pred_val = outputs[pred_key]
                target_val = outputs[target_key]

                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)

                l = w * l
                lower_loss += l
                loss_dict[f"lower_{pred_key}_{loss_fn.__class__.__name__}"] = l

        # 反向传播
        self.amp_scaler.scale(lower_loss).backward()
        self.amp_scaler.unscale_(self.optimizer_i)

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(lower_params, 0.01)

        # 🔥 4. 执行下层优化：w_new = w_old - lr * g_lower
        self.amp_scaler.step(self.optimizer_i)
        self.amp_scaler.update()

        # 🔥 4.5 保存优化后的R参数（w_new）
        r_params_new = [p.data.clone() for p in r_params]

        # ==================== 阶段2: 上层优化（事件分支 E）====================

        # 🔥 5. 计算上层损失（在 w_new 位置）
        self.net_g.zero_grad()

        with autocast('cuda', enabled=self.use_amp):
            preds_upper, I_upper, R_upper, I_gt_upper, R_gt_upper, event_upper = \
                self.net_g(self.lq, self.event, self.gt)

            pred_label = self.eq24_eq25_improved(self.event, (R_gt_upper - R_upper), 0.01)

            pred_label[pred_label > 0] = 1  # pos
            pred_label[pred_label < 0] = 2  # neg
            pred_label = pred_label.long()

            outputs_upper = {
                "preds": preds_upper[-1] if isinstance(preds_upper, list) else preds_upper,
                "I": I_upper,
                "R": R_upper,
                "gt": self.gt,
                "I_gt": I_gt_upper,
                "R_gt": R_gt_upper,
                "pred_voxel": event_upper,
                "mask_label1": pred_label,
                "mask_label": self.mask
            }

            # 上层损失 = E自身的事件损失 + E增强R后的图像质量
            upper_loss = 0.0
            event_loss = 0.0
            image_quality_loss = 0.0

            # (1) E自身的事件重建损失
            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_voxel_list, self.cri_voxel_weight,
                    self.cri_voxel_pred_key, self.cri_voxel_target_key):

                pred_val = outputs_upper[pred_key]
                target_val = outputs_upper[target_key]

                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)

                l = w * l
                event_loss += l
                loss_dict[f"upper_event_{pred_key}_{loss_fn.__class__.__name__}"] = l

            # (2) E增强R后的图像质量
            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_r_related_list, self.cri_r_related_weight,
                    self.cri_r_related_pred_key, self.cri_r_related_target_key):

                pred_val = outputs_upper[pred_key]
                target_val = outputs_upper[target_key]

                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)

                l = w * l
                image_quality_loss += l
                loss_dict[f"upper_img_{pred_key}_{loss_fn.__class__.__name__}"] = l

            # 组合上层损失
            event_loss_weight = train_opt.get('event_loss_weight', 1.0)
            image_loss_weight = train_opt.get('upper_image_weight', 1.0)

            upper_loss = event_loss_weight * event_loss + image_loss_weight * image_quality_loss

            loss_dict['event_loss'] = event_loss
            loss_dict['image_quality_loss'] = image_quality_loss

        # 🔥 6. 计算上层损失对上层参数的直接梯度
        self.amp_scaler.scale(upper_loss).backward()
        self.amp_scaler.unscale_(self.optimizer_e)

        # 保存E参数的直接梯度
        upper_params = []
        upper_direct_grads = []

        for name, p in self.net_g.named_parameters():
            if 'Enet' in name:
                upper_params.append(p)
                if p.grad is None:
                    upper_direct_grads.append(torch.zeros_like(p))
                else:
                    upper_direct_grads.append(p.grad.data.clone())

        # 取上层反向时关于R的梯度 ∇_R L_upper
        vector = []
        for p in r_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
            vector.append(p.grad.data.clone())

        # 🔥 8. 计算隐式梯度
        epsilon = train_opt.get('bilevel_epsilon', 1e-2)
        lr_lower = self.optimizer_i.param_groups[0]['lr']

        grad_norm = torch.cat([g.reshape(-1) for g in vector]).norm()
        R_step = epsilon / (grad_norm + 1e-8)

        # 🔥 9. (1) w+ = w_old + R * vector（只扰动R）
        for p, p_old, v in zip(r_params, r_params_old, vector):
            p.data.copy_(p_old)
            p.data.add_(v, alpha=R_step)

        # 计算W+状态下的梯度
        self.net_g.zero_grad()

        with autocast('cuda', enabled=self.use_amp):
            preds_p, I_p, R_p, I_gt_p, R_gt_p, event_p = self.net_g(self.lq, self.event, self.gt)

            lower_loss_p = 0.0
            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_r_related_list, self.cri_r_related_weight,
                    self.cri_r_related_pred_key, self.cri_r_related_target_key):
                pred_val = preds_p[-1] if isinstance(preds_p, list) else preds_p if pred_key == 'preds' else (
                    I_p if pred_key == 'I' else R_p
                )
                target_val = self.gt if target_key == 'gt' else (
                    I_gt_p if target_key == 'I_gt' else R_gt_p
                )
                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)
                lower_loss_p += w * l

        self.amp_scaler.scale(lower_loss_p).backward()
        self.amp_scaler.unscale_(self.optimizer_e)

        # 得到上层网络的梯度
        grads_p = []
        for name, p in self.net_g.named_parameters():
            if 'Enet' in name:
                if p.grad is None:
                    grads_p.append(torch.zeros_like(p))
                else:
                    grads_p.append(p.grad.data.clone())

        # 🔥 10. (2) w- = w_old - R * vector
        for p, p_old, v in zip(r_params, r_params_old, vector):
            p.data.copy_(p_old)
            p.data.sub_(v, alpha=R_step)

        self.net_g.zero_grad()

        with autocast('cuda', enabled=self.use_amp):
            preds_n, I_n, R_n, I_gt_n, R_gt_n, event_n = self.net_g(self.lq, self.event, self.gt)

            lower_loss_n = 0.0
            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_r_related_list, self.cri_r_related_weight,
                    self.cri_r_related_pred_key, self.cri_r_related_target_key):
                pred_val = preds_n[-1] if isinstance(preds_n, list) else preds_n if pred_key == 'preds' else (
                    I_n if pred_key == 'I' else R_n
                )
                target_val = self.gt if target_key == 'gt' else (
                    I_gt_n if target_key == 'I_gt' else R_gt_n
                )
                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)
                lower_loss_n += w * l

        self.amp_scaler.scale(lower_loss_n).backward()
        self.amp_scaler.unscale_(self.optimizer_e)

        grads_n = []
        for name, p in self.net_g.named_parameters():
            if 'Enet' in name:
                if p.grad is None:
                    grads_n.append(torch.zeros_like(p))
                else:
                    grads_n.append(p.grad.data.clone())

        # 🔥 11. 恢复R参数到优化后的状态（w_new）
        for p, p_new in zip(r_params, r_params_new):
            p.data.copy_(p_new)

        # 🔥 12. 计算隐式梯度
        implicit_grads = []
        for gp, gn in zip(grads_p, grads_n):
            implicit_grad = -lr_lower * (gp - gn) / (2 * R_step)
            implicit_grads.append(implicit_grad)

        # 🔥 13. 组合梯度
        final_grads = []
        for direct_grad, implicit_grad in zip(upper_direct_grads, implicit_grads):
            final_grads.append(direct_grad + implicit_grad)

        # 🔥 14. 赋值梯度
        idx = 0
        for name, p in self.net_g.named_parameters():
            if 'Enet' in name:
                if p.grad is None:
                    p.grad = final_grads[idx]
                else:
                    p.grad.data.copy_(final_grads[idx])
                idx += 1

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(upper_params, 0.01)

        # 🔥 15. 更新上层参数
        self.amp_scaler.step(self.optimizer_e)
        self.amp_scaler.update()

        # ==================== 记录和EMA ====================
        self.output = outputs["preds"]
        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict['lower_loss'] = lower_loss
        self.log_dict['upper_loss'] = upper_loss

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # for update_img in [True, False]:
        #
        #
        #     optimizer = self.optimizer_i if update_img else self.optimizer_e
        #     optimizer.zero_grad()
        #
        #     with autocast(enabled=self.use_amp):
        #         self.net_g.mode='train'
        #         preds, I, R, I_gt, R_gt,event = self.net_g(self.lq, self.gt,self.event)  # 修改：模型返回多分支
        #         outputs = {
        #             "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
        #             "I": I,
        #             "R": R,
        #             "gt": self.gt,  # 最终GT
        #             "I_gt": I_gt,  # 预分解光照GT
        #             "R_gt": R_gt,  # 预分解反射率GT
        #             "pred_voxel": event,
        #             "event_gt": self.event_gt
        #         }  # 新增：统一输出字典
        #
        #     # if not isinstance(preds, list):
        #     #         preds = [preds]
        #
        #         #self.output = preds[-1] if isinstance(preds, list) else preds
        #
        #         if update_img:
        #             cri_list = self.cri_pix_list
        #             cri_weight = self.cri_pix_weight
        #             cri_pred_key = self.cri_pix_pred_key
        #             cri_target_key = self.cri_pix_target_key
        #         else:
        #             cri_list = self.cri_voxel_list
        #             cri_weight = self.cri_voxel_weight
        #             cri_pred_key = self.cri_voxel_pred_key
        #             cri_target_key = self.cri_voxel_target_key
        #
        #         loss_dict = OrderedDict()
        #         total_loss = 0.0  # 修改：使用 total_loss 累加多个 loss
        #
        #         # ---------------- 修改 ----------------
        #         for loss_fn, w, pred_key, target_key in zip(
        #                 cri_list, cri_weight, cri_pred_key, cri_target_key):
        #
        #             pred_val = outputs[pred_key]
        #             target_val = outputs[target_key]
        #
        #             # 如果 pred_val 是 list，则累加每个元素
        #             if isinstance(pred_val, list):
        #                 l = sum([loss_fn(p, target_val) for p in pred_val])
        #             else:
        #                 l = loss_fn(pred_val, target_val)
        #
        #             l = w * l  # 修改：加权
        #             total_loss += l
        #             loss_dict[f"{pred_key}_{loss_fn.__class__.__name__}"] = l  # 修改：保存每个 loss
        #
        #         self.output = outputs["preds"]  # 修改：输出最终预测结果
        #
        #         # ---------------- 修改 ----------------
        #     self.amp_scaler.scale(total_loss).backward()  # 混合精度
        #     self.amp_scaler.unscale_(optimizer)  # 保留原逻辑
        #
        #     # 验证梯度
        #     if current_iter % 100 == 0:  # 每100次打印一次
        #         logger = get_root_logger()
        #         if update_img:
        #             logger.info("=== Optimizing Image Branch ===")
        #             # 检查 E 分支是否有梯度（应该有，但不更新）
        #             for name, p in self.net_g.named_parameters():
        #                 if 'embedding_e' in name and p.grad is not None:
        #                     logger.info(f"E branch {name} has grad: {p.grad.norm().item():.6f}")
        #                     break
        #         else:
        #             logger.info("=== Optimizing Event Branch ===")
        #             # 检查 R 分支是否有梯度（应该有，但不更新）
        #             for name, p in self.net_g.named_parameters():
        #                 if 'embedding_r' in name and p.grad is not None:
        #                     logger.info(f"R branch {name} has grad: {p.grad.norm().item():.6f}")
        #                     break
        #     # l_pix.backward()
        #
        #     if self.opt['train']['use_grad_clip']:
        #         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        #     # self.optimizer_g.step()




    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img_gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        event_pad = F.pad(self.event, (0, mod_pad_w, 0, mod_pad_h), mode='constant', value=0)

        self.nonpad_test(img,event_pad,img_gt)
        _, _,h, w = self.output.size()
        self.output = self.output[:, :,0:h -mod_pad_h * scale, 0:w - mod_pad_w * scale]
       # self.outputs["R_gt"] = self.outputs["R_gt"][:, :,0:h -mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.outputs["R"] = self.outputs["R"][:, :,0:h -mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.outputs["pred_voxel"] = self.outputs["pred_voxel"][:,:, :,0:h -mod_pad_h * scale, 0:w - mod_pad_w * scale]
       # pred_label = self.eq24_eq25_improved(self.event, (self.outputs["R_gt"] - self.outputs["R"]), 0.01)
       # pred_label[pred_label > 0] = 1  # pos
       # pred_label[pred_label < 0] = 2  # neg
       # pred_label = pred_label.long()
        #self.outputs["mask_label1"] = pred_label
    def nonpad_test(self, img=None,event_low=None,img_gt=None):
        self.net_g.mode='test'
        #event_de = self.event_denoising
        if img is None:
            img = self.lq
            img_gt = self.gt
            event_low = self.event
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                preds, I, R,event = self.net_g(img,event_low,img_gt)  # 修改：模型返回多分支

                #
                # pred_label = self.eq24_eq25_improved(self.event,(R_gt-R),0.01)
                #
                # pred_label[pred_label > 0] = 1  # pos
                # pred_label[pred_label < 0] = 2  # neg
                # pred_label = pred_label.long()

                outputs = {
                    "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
                    "I": I,
                    "R": R,
                    "gt": img_gt,  # 最终GT
                    "pred_voxel": event,
                    #"mask_label1": pred_label,
                   # "mask_label": event_de
                }  # 新增：统一输出字典
            # if isinstance(pred, list):
            #     pred = pred[-1]
            self.output = preds
            self.outputs = outputs
            self.net_g_ema.train()
        else:
            self.net_g.eval()
            with torch.no_grad():
                preds, I, R, event = self.net_g(img, event_low,img_gt)  # 修改：模型返回多分支

                # pred_label = self.eq24_eq25_improved(self.event, (R_gt - R), 0.01)
                # pred_label[pred_label > 0] = 1  # pos
                # pred_label[pred_label < 0] = 2  # neg
                # pred_label = pred_label.long()
                outputs = {
                    "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
                    "I": I,
                    "R": R,
                    "gt": img_gt,  # 最终GT
                    "pred_voxel": event,
                    #"mask_label1": pred_label,
                    #"mask_label": event_de

                }  # 新增：统一输出字典
            self.output = preds
            self.outputs = outputs
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']  #self.net_g.mode='val'
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        save_cfg = self.opt.get('val', {})
        save_images = save_cfg.get('save_images', True)
        save_first_n = int(save_cfg.get('save_first_n', 0))  # 0/负数=不限制

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            event = self.event[:1]  # .squeeze(dim=0)
            event = event.squeeze(dim=0)

           # event_denoising = self.event_denoising[:1]
            #event_denoising = event_denoising.squeeze(dim=0)

            height, width = event.shape[1:]
           # mask_gt =  self.mask[:1]
           # mask_gt[mask_gt == 2] = -1  # 转化方便可视化
           # mask_gt = mask_gt.squeeze(dim=0)
           # pred_mask =  self.outputs['mask_label1'][:1]

           # pred_mask[pred_mask == 2] = -1
           # pred_mask = pred_mask.squeeze(dim=0)
             #self.outputs['pred_voxel'].detach().cpu()

           # pred_event = self.outputs['pred_voxel']


            visuals = self.get_current_visuals()
            pred_event = visuals['Event'][:1]
            pred_event = pred_event.squeeze(dim=0)
            sr_img = tensor2img([visuals['result'][:1]], rgb2bgr=rgb2bgr)
            sr_R = tensor2img([visuals['R'][:1]], rgb2bgr=rgb2bgr)
            sr_I = tensor2img([visuals['I'][:1]], rgb2bgr=rgb2bgr)
            gt_img = None
            # if 'gt' in visuals:
            #     gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
            #     del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()


            # ---------- 只保存前 N 张 ----------
            do_save = bool(save_images and save_img and (save_first_n <= 0 or idx < save_first_n))
            if do_save:
                if self.opt['is_train']:
                    base_dir = osp.join(self.opt['path']['visualization'], img_name)
                    save_img_path = osp.join(base_dir, f'{img_name}_{current_iter}.png')
                    save_gt_img_path = osp.join(base_dir, f'{img_name}_{current_iter}_gt.png')
                    save_imgI_path = osp.join(base_dir, f'{img_name}_{current_iter}_I.png')
                    save_imgR_path = osp.join(base_dir, f'{img_name}_{current_iter}_R.png')
                    #save_event_path = osp.join(base_dir, f'{img_name}_{current_iter}_Event.png')

                    save_maskgt_path = osp.join(base_dir, f'{img_name}_{current_iter}_maskgt.png')
                    save_pred_mask_path = osp.join(base_dir, f'{img_name}_{current_iter}_pred_mask.png')
                    save_event_path = osp.join(base_dir, f'{img_name}_{current_iter}_Event_low.png')
                    save_pred_event_path = osp.join(base_dir, f'{img_name}_{current_iter}_pred_event.png')

                else:
                    base_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    save_img_path = osp.join(base_dir, f'{img_name}.png')
                    save_gt_img_path = osp.join(base_dir, f'{img_name}_gt.png')
                    save_imgI_path = osp.join(base_dir, f'{img_name}_I.png')
                    save_imgR_path = osp.join(base_dir, f'{img_name}_R.png')
                   # save_maskgt_path = osp.join(base_dir, f'{img_name}_{current_iter}_maskgt.png')
                   # save_pred_mask_path = osp.join(base_dir, f'{img_name}_{current_iter}_pred_mask.png')
                    save_event_path = osp.join(base_dir, f'{img_name}_{current_iter}_Event_low.png')
                    save_pred_event_path = osp.join(base_dir, f'{img_name}_{current_iter}_pred_event.png')
                os.makedirs(base_dir, exist_ok=True)
                imwrite(sr_img, save_img_path)
                imwrite(sr_I, save_imgI_path)
                imwrite(sr_R, save_imgR_path)


                #visualize_event_image(mask_gt, height, width, saveimg=True, savepath=save_maskgt_path)
                #visualize_event_image(pred_mask, height, width, saveimg=True, savepath=save_pred_mask_path)
                visualize_event_image(event, height, width, saveimg=True, savepath=save_event_path)
                visualize_event_image(pred_event, height, width, saveimg=True, savepath=save_pred_event_path)
                if gt_img is not None:
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:

                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += self.m_psnr(visuals['result'], visuals['gt'])

                            # getattr(
                            # metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['R'] = self.outputs['R'].detach().cpu()
        out_dict['I'] = self.outputs['I'].detach().cpu()

        pred_vis =self.outputs['pred_voxel'].argmax(dim=1)
        pred_vis[pred_vis == 2] = -1

        out_dict['Event'] = pred_vis #self.outputs['pred_voxel'].detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
