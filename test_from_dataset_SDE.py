import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from skimage.util import img_as_ubyte
from torch.utils.data import DataLoader
from tqdm import tqdm

import Enhancement.utils as utils
from basicsr.models import create_model
from basicsr.utils.options import parse


def visualize_event_image(voxel_grid, height, width, savepath=None, saveimg=False, show=False):
    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    full_frame = np.zeros((height, width), dtype=np.float32)
    for t_bin in range(voxel_grid.shape[0]):
        full_frame += voxel_grid[t_bin]

    binary_frame = np.sign(full_frame)
    rgb_image = np.ones((height, width, 3), dtype=np.float32)
    rgb_image[binary_frame > 0] = [1, 0, 0]
    rgb_image[binary_frame < 0] = [0, 0, 1]

    if saveimg:
        plt.imsave(savepath, rgb_image)
    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title('Binary Event Visualization')
        plt.show()


def load_weights(model, weights):
    checkpoint = torch.load(weights)
    state = checkpoint['params'] if 'params' in checkpoint else checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for key, value in state.items():
            if key.startswith('module.'):
                new_state[key.replace('module.', '', 1)] = value
            else:
                new_state['module.' + key] = value
        model.load_state_dict(new_state)


def main():
    parser = argparse.ArgumentParser(description='BiEvLight SDE dataset inference')
    parser.add_argument('--result_dir', default='./results/', type=str)
    parser.add_argument('--opt', type=str, default='Options/BiEvLight_inference.yml')
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--dataset', default='SDE', type=str)
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print('export CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'])
    print(f'dataset {args.dataset}')

    opt = parse(args.opt, is_train=False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g
    weights = args.weights or opt['path'].get('pretrain_network_g')
    if weights is None:
        raise ValueError('Please set --weights or path.pretrain_network_g in the YAML.')
    load_weights(model_restoration, weights)

    print('===>Testing using weights: ', weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    factor = 4
    config = os.path.basename(args.opt).split('.')[0]
    checkpoint_name = os.path.basename(weights).split('.')[0]
    result_dir = os.path.join(args.result_dir, args.dataset, config, checkpoint_name)
    result_dir_L = os.path.join(result_dir, 'L')
    result_dir_R = os.path.join(result_dir, 'R')
    result_dir_event = os.path.join(result_dir, 'Event')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir_event, exist_ok=True)
    os.makedirs(result_dir_L, exist_ok=True)
    os.makedirs(result_dir_R, exist_ok=True)

    from basicsr.data.SDE_event_bilievel_dataset import Bilevel_eventDataset as Dataset

    dataset_opt = opt['datasets']['val']
    dataset_opt['phase'] = 'test'
    if dataset_opt.get('scale') is None:
        dataset_opt['scale'] = 1

    dataset = Dataset(dataset_opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    psnr = []
    ssim = []
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            gt = data_batch['gt']
            target = gt.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            h, w = input_.shape[2], input_.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            event = data_batch['event']
            event_pad = F.pad(event, (0, padw, 0, padh), mode='constant', value=0)

            restored, L, R, I_gt, R_gt, event_logits = model_restoration(input_, event_pad, gt)

            restored = restored[:, :, :h, :w]
            L = L[:, :, :h, :w]
            R = R[:, :, :h, :w]
            restored_np = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(target, restored_np))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored_np)))

            type_id = os.path.basename(os.path.dirname(inp_path))
            base = os.path.splitext(os.path.basename(inp_path))[0]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)

            utils.save_img(str(Path(result_dir, type_id, f'{base}.png')), img_as_ubyte(restored_np))

            event_vis = event_logits.argmax(dim=1)
            event_vis[event_vis == 2] = -1
            event_vis = event_vis.squeeze(dim=0)
            height, width = event_vis.shape[1:]
            visualize_event_image(
                event_vis,
                height,
                width,
                saveimg=True,
                savepath=os.path.join(result_dir_event, f'{base}_event.png'))
            torchvision.utils.save_image(L, os.path.join(result_dir_L, f'{base}_L.png'))
            torchvision.utils.save_image(R, os.path.join(result_dir_R, f'{base}_R.png'))

    print('PSNR: %f ' % float(np.mean(np.array(psnr))))
    print('SSIM: %f ' % float(np.mean(np.array(ssim))))


if __name__ == '__main__':
    main()
