# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

from ast import arg
import numpy as np
import os
import argparse

import torchvision.utils
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import Enhancement.utils as utils

from natsort import natsorted
from glob import glob
from skimage.util import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from pathlib import Path
import matplotlib.pyplot as plt

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")


def visualize_event_image(voxel_grid, height, width, savepath=None, saveimg=False, show=False):
    """功能1：生成累积的事件图像（空间投影图）- 二值化颜色"""
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    full_frame = np.zeros((height, width), dtype=np.float32)
    for t_bin in range(voxel_grid.shape[0]):
        full_frame += voxel_grid[t_bin]

    # 二值化：只保留正负符号
    binary_frame = np.sign(full_frame)  # 转换为 -1, 0, 1

    # 创建RGB图像：红色表示正事件，蓝色表示负事件
    rgb_image = np.ones((height, width, 3), dtype=np.float32)  # 白色背景

    # 正事件 -> 红色 (1, 0, 0)
    rgb_image[binary_frame > 0] = [1, 0, 0]

    # 负事件 -> 蓝色 (0, 0, 1)
    rgb_image[binary_frame < 0] = [0, 0, 1]

    # 零值保持白色 (1, 1, 1) - 已经是默认值

    if saveimg:
        plt.imsave(savepath, rgb_image)

    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title('Binary Event Visualization')
        plt.show()
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_L  = os.path.join(args.result_dir, dataset, config, checkpoint_name,'L')
result_dir_R  = os.path.join(args.result_dir, dataset, config, checkpoint_name,'R')
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
result_dir_gt_L = os.path.join(args.result_dir, dataset, 'gt','L')
result_dir_gt_R = os.path.join(args.result_dir, dataset, 'gt','R')
result_dir_event =  os.path.join(args.result_dir, dataset, config, checkpoint_name,'Event')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_dir_event, exist_ok=True)
os.makedirs(result_dir_L, exist_ok=True)
os.makedirs(result_dir_R, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []

if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor','SDE','SDSD']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    os.makedirs(result_dir_gt_L, exist_ok=True)
    os.makedirs(result_dir_gt_R, exist_ok=True)
    # if dataset == 'SID':
    #     from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    # elif dataset == 'SMID':
    #     from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    # elif dataset =='SDE':
    #     from basicsr.data.SDE_image_dataset import Dataset_PairedImage_SDE as Dataset
    # else:
    #     from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    #from basicsr.data.SDE_event_bilievel_dataset import Bilevel_eventDataset as Dataset
    from basicsr.data.SDSD_event_bilievel_dataset import SDSD_Bilevel_eventDataset as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    # if '~' in opt['dataroot_gt']:
    #     opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    # if '~' in opt['dataroot_lq']:
    #     opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            gt = data_batch['gt']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            event = data_batch['event']
            event_pad = F.pad(event, (0, padw, 0, padh), mode='constant', value=0)

            if args.self_ensemble:
                restored, I, R, I_gt, R_gt, event = self_ensemble(input_, model_restoration)
            else:
                restored, L, R, I_gt, R_gt,event = model_restoration(input_,event_pad,gt)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            L = L[:, :, :h, :w]
            R = R[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            type_id = os.path.basename(os.path.dirname(inp_path))

            event = event.argmax(dim=1)
            event[event == 2] = -1
            event=event.squeeze(dim=0)
            height, width = event.shape[1:]

            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            # --- 保存：模型复原、输入、GT（用你原来的 utils + img_as_ubyte）---
            base = os.path.splitext(os.path.basename(inp_path))[0]
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(inp_path)))#[2]
            utils.save_img(str(Path(result_dir, type_id, f"{folder_name}_{base}.png")), img_as_ubyte(restored))

            visualize_event_image(event, height, width, saveimg=True, savepath=os.path.join(result_dir_event, f"{folder_name}_{base}_event.png"))
            torchvision.utils.save_image(L, os.path.join(result_dir_L, f"{folder_name}_{base}_L.png"))
            torchvision.utils.save_image(R, os.path.join(result_dir_R,  f"{folder_name}_{base}_R.png"))
            #torchvision.utils.save_image(L_1, os.path.join(result_dir_gt_L, f"{base}_L1.png"))
            #torchvision.utils.save_image(R_1, os.path.join(result_dir_gt_R, f"{base}_R1.png"))

else:

    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

    target_paths = natsorted(glob(os.path.join(
        target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)
            else:
                # split and test
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
