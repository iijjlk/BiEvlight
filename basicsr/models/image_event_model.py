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

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


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


class ImageCleanModel_Retinex(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel_Retinex, self).__init__(opt)

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
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

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

        # define losses
        if train_opt.get('pixel_opt_list'):
            # pixel_type = train_opt['pixel_opt'].pop('type')
            # cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            # self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
            #     self.device)      #如何写 weighted loss 呢？传参构造Loss函数
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

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = data['lq'].to(self.device)
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        with autocast(enabled=self.use_amp):
            self.net_g.mode='train'
            preds, I, R, I_gt, R_gt = self.net_g(self.lq, self.gt)  # 修改：模型返回多分支
            outputs = {
                "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
                "I": I,
                "R": R,
                "gt": self.gt,  # 最终GT
                "I_gt": I_gt,  # 预分解光照GT
                "R_gt": R_gt  # 预分解反射率GT
            }  # 新增：统一输出字典

        # if not isinstance(preds, list):
        #         preds = [preds]

            #self.output = preds[-1] if isinstance(preds, list) else preds

            loss_dict = OrderedDict()
            total_loss = 0.0  # 修改：使用 total_loss 累加多个 loss

            # ---------------- 修改 ----------------
            for loss_fn, w, pred_key, target_key in zip(
                    self.cri_pix_list, self.cri_pix_weight,
                    self.cri_pix_pred_key, self.cri_pix_target_key):

                pred_val = outputs[pred_key]
                target_val = outputs[target_key]

                # 如果 pred_val 是 list，则累加每个元素
                if isinstance(pred_val, list):
                    l = sum([loss_fn(p, target_val) for p in pred_val])
                else:
                    l = loss_fn(pred_val, target_val)

                l = w * l  # 修改：加权
                total_loss += l
                loss_dict[f"{pred_key}_{loss_fn.__class__.__name__}"] = l  # 修改：保存每个 loss

            self.output = outputs["preds"]  # 修改：输出最终预测结果

            # ---------------- 修改 ----------------
        self.amp_scaler.scale(total_loss).backward()  # 修改：用 total_loss 反向传播
        self.amp_scaler.unscale_(self.optimizer_g)  # 保留原逻辑
        # l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # self.optimizer_g.step()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

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
        self.nonpad_test(img,img_gt)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None,img_gt=None):
        self.net_g.mode='test'
        if img is None:
            img = self.lq
            img_gt = self.gt
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                preds, I, R, I_gt, R_gt = self.net_g(img, img_gt)  # 修改：模型返回多分支
                outputs = {
                    "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
                    "I": I,
                    "R": R,
                    "gt": img_gt,  # 最终GT
                    "I_gt": I_gt,  # 预分解光照GT
                    "R_gt": R_gt  # 预分解反射率GT
                }  # 新增：统一输出字典
            # if isinstance(pred, list):
            #     pred = pred[-1]
            self.output = preds
            self.outputs = outputs
        else:
            self.net_g.eval()
            with torch.no_grad():
                preds, I, R, I_gt, R_gt = self.net_g(img, img_gt)  # 修改：模型返回多分支
                outputs = {
                    "preds": preds[-1] if isinstance(preds, list) else preds,  # 最终输出
                    "I": I,
                    "R": R,
                    "gt": img_gt,  # 最终GT
                    "I_gt": I_gt,  # 预分解光照GT
                    "R_gt": R_gt  # 预分解反射率GT
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

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            sr_R = tensor2img([visuals['R']], rgb2bgr=rgb2bgr)
            sr_I = tensor2img([visuals['I']], rgb2bgr=rgb2bgr)
            gt_img = None
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

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
                else:
                    base_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    save_img_path = osp.join(base_dir, f'{img_name}.png')
                    save_gt_img_path = osp.join(base_dir, f'{img_name}_gt.png')
                    save_imgI_path = osp.join(base_dir, f'{img_name}_I.png')
                    save_imgR_path = osp.join(base_dir, f'{img_name}_R.png')

                os.makedirs(base_dir, exist_ok=True)
                imwrite(sr_img, save_img_path)
                imwrite(sr_I, save_imgI_path)
                imwrite(sr_R, save_imgR_path)
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
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

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
