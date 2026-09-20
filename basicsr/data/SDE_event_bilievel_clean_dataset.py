

from torch.utils import data as data
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_SDE_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                     paired_paths_from_folder,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import *
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP,imfrombytesDP

import random
import numpy as np
import torch
import cv2
from pdb import set_trace as stx
from pathlib import Path
import matplotlib.pyplot as plt
#from  basicsr.models.archs.eventformer_Bilevel_arch import Decom,load_initialize
import os
import torch.nn as nn
def load_initialize(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path)

        model.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # to freeze the params of Decomposition Model
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()
#Pretrained RetinexNet
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,dilation=1,groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,dilation=1,groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,dilation=1,groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input.cuda())
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R.detach().cpu(), L

def spatial_gradient_2d(voxel):
    # voxel: (B, C, H, W)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=voxel.dtype, device=voxel.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=voxel.dtype, device=voxel.device)

    sobel_x = sobel_x.view(1,1,3,3).repeat(voxel.size(1),1,1,1)  # (C,1,3,3)
    sobel_y = sobel_y.view(1,1,3,3).repeat(voxel.size(1),1,1,1)

    grad_x = F.conv2d(voxel, sobel_x, padding=1, groups=voxel.size(1))
    grad_y = F.conv2d(voxel, sobel_y, padding=1, groups=voxel.size(1))

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad_mag, grad_x, grad_y

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



class Bilevel_eventDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Bilevel_eventDataset, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.is_split_event = opt['is_split_event']
        self.whole_low_event_npz = None#opt['whole_low_event_npz']
        self.whole_normal_event_npz = None#opt['whole_normal_event_npz']

        # self.Decom =  Decom().cuda()
        # self.Decom = load_initialize(self.Decom,'G:\work\code\low_level\LLIE\Retinexformer\ckpt\init_low.pth')#--------------------------base params-----------------------------
        self.H = opt['height']
        self.W = opt['width']
       # self.center_cropped_height = opt['gt_size']
        #self.random_cropped_width = opt['gt_size']
        self.seq_name = "merged"
        self.voxel_channel = opt['voxel_channel']
        self.is_split_event = opt['is_split_event']


        self.dataset_root = opt['dataset_root']
        root = Path(self.dataset_root)
        self.gt_folder = root / opt['normal_subdir']
        self.lq_folder = root/ opt['low_subdir']
        self.event_folder = root / opt['events_subdir']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'


        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']


        #处理数据字段
        low_pngs = sorted([p for p in self.lq_folder.iterdir() if p.suffix.lower() == '.png'])
        normal_pngs = sorted([p for p in self.gt_folder.iterdir() if p.suffix.lower() == '.png'])
        n = min(len(low_pngs), len(normal_pngs))

        if n == 0:
            raise RuntimeError(f'No png files under {self.lq_folder} or {self.gt_folder}')
        if len(low_pngs) != len(normal_pngs):
            print(f"[warn] low({len(low_pngs)}) != normal({len(normal_pngs)}); pairing first {n} by index")

        # ==== CHANGED: 直接保存完整路径与文件名 ====
        self.low_img_paths = [str(p) for p in low_pngs[:n]]
        self.normal_img_paths = [str(p) for p in normal_pngs[:n]]

        self.low_img_list = [p.name for p in low_pngs[:n]]                                        #only name
        self.normal_img_list = [p.name for p in normal_pngs[:n]]

        if self.is_split_event:
            self.ev_paths = []
            self.ev_normal_paths = []
            self.ev_denoise_paths= []
            self.ev_denoise_list = []
            self.ev_list = []  # events_name
            self.ev_normal_list = []

            for lp, np in zip(low_pngs[:n], normal_pngs[:n]):
                    # 生成低光图像和正常光图像对应的事件文件名（去掉 .png 后缀，替换为 .npz）
                    event_name = lp.with_suffix('.npz').name

                    # 查找低光图像的事件文件
                    ev_path = self.event_folder / event_name
                    if not ev_path.exists():
                        raise FileNotFoundError(
                            f"Event file for low light image {event_name} not found in {self.event_folder}")
                    self.ev_paths.append(str(ev_path))
                    self.ev_list.append(ev_path.name)

                    # 粗去噪事件


        else:
            # whole-event 模式（合并后通常不建议用；保留兼容）
            self.low_event_file = (
                str(self.root / "lowlight_event.npz") if self.whole_low_event_npz is None else self.whole_low_event_npz
            )
            self.events_denoising_file = (
                str(self.root / "normalight_event.npz") if self.whole_normal_event_npz is None else self.whole_normal_event_npz
            )
        self.num_input = n
        self.ev_idx = None
        self.events = None

    def __len__(self):
        return self.num_input


    def get_event(self, idx):
        """Split single event from the whole event file using timestamps from filenames."""
        # 文件名可能不是纯时间，这里按原逻辑：low_img_list 保存的是文件名，
        # 取上一张/下一张文件名去掉后缀作为时间戳；若不全是数字会报错——和你原先一致
        if idx == 0:
            start_t = self.events[0, 0]
        else:
            start_t = int(Path(self.low_img_list[idx - 1]).stem)

        if idx == self.num_input - 1:
            end_t = self.events[-1, 0]
        else:
            end_t = int(Path(self.low_img_list[idx + 1]).stem)

        ev_start_idx = np.where(self.events[:, 0] > start_t)[0][0]
        ev_end_idx = np.where(self.events[:, 0] < end_t)[0][-1]
        return self.events[ev_start_idx:ev_end_idx]

    def _crop(self, input_frame_list, events_list):
        """crop frame and events
           training: random crop
           testing: central crop
        """
        if self.is_train:
            min_y = random.randint(0, self.W - self.random_cropped_width)
            min_x = random.randint(0, self.H - self.center_cropped_height)
        else:
            min_y = (self.W - self.random_cropped_width) // 2
            min_x = (self.H - self.center_cropped_height) // 2

        max_y = min_y + self.random_cropped_width
        max_x = min_x + self.center_cropped_height

        crop_image_list = []
        for input_frame in input_frame_list:
            input_frames = input_frame[min_x:max_x, min_y:max_y, :]
            input_frames_torch = torch.from_numpy(input_frames).permute(2, 0, 1).float()
            crop_image_list.append(input_frames_torch)

        output_events_list = []
        for events in events_list:
            mask_x = torch.where((events[:, 2] < max_x) & (events[:, 2] >= min_x))
            event_x = torch.index_select(events, 0, mask_x[0])
            mask_y = torch.where((event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y))
            event_y = torch.index_select(event_x, 0, mask_y[0])
            event = event_y.clone()
            event[:, 2] = event_y[:, 2] - min_x
            event[:, 1] = event_y[:, 1] - min_y
            output_events_list.append(event)
        return crop_image_list, output_events_list

    def _generate_voxel_grid(self, event, height=None, width=None):
        """obtain voxel grid """
        if event is None or event.numel() == 0:
            device = event.device
            return torch.zeros(
                (self.voxel_channel, height, width),
                dtype=torch.float32,
                device=device
            )
        event_start = event[0, 0]
        event_end = event[-1, 0]

        ch = (event[:, 0].to(torch.float32) / (event_end - event_start) * self.voxel_channel).long()
        torch.clamp_(ch, 0, self.voxel_channel - 1)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1

        voxel_grid = torch.zeros((self.voxel_channel, height, width), dtype=torch.float32)
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)
        return voxel_grid

    def events_to_image(self, xs,ys,ps,sensor_size=(260, 346)):
        """
        Accumulate events into an image.
        xs, ys, ps: torch.tensor, [N]
        """
        # xs = xs - 1
        # ys = ys - 1
        xs_mask = (xs >= sensor_size[1]) + (xs < 0)
        ys_mask = (ys >= sensor_size[0]) + (ys < 0)
        mask = xs_mask + ys_mask
        xs[mask] = 0
        ys[mask] = 0
        ps[mask] = 0

        device = xs.device
        img_size = list(sensor_size)
        img = torch.zeros(img_size).to(device)

        xs = xs.to(torch.float32)  # Converting to float32 for computations if necessary

        ps = ps.to(torch.float32)
        ys = sensor_size[0] - ys - 1
        ys = ys.to(torch.float32)

        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)

        img.index_put_((ys, xs), ps, accumulate=True)

        return img

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # ------------------load event_flow--------------------
        if (self.events is None) and (self.is_split_event == False):
            events = np.load(self.low_event_file)
            events_denoising = np.load(self.events_denoising_file)
        elif self.is_split_event == True:
            ev_path = self.ev_paths[index]

            if ev_path is None:
                raise FileNotFoundError(f"No split event npz for low frame: {self.low_img_list[index]}")
            events = np.load(ev_path)
        else:
            raise ValueError('w/o assign event')

        try:
            self.events = events["arr_0"] if "arr_0" in events else events  #[t,x,y,p]
            #self.events_normal = events_normal#["arr_0"] if "arr_0" in events_normal else events_normal
            if self.events.ndim == 1:  # 结构化数组
                et = self.events["timestamp"]
                ex = self.events["x"]
                ey = self.events["y"]
                ep = self.events["polarity"]
                self.events = np.stack([et, ex, ey, ep], axis=1)
        except Exception as e:
            print(f"loading event error @ index: {index} ({e})")

        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
            except Exception as e:
                print(f"loading event error @ seq: {self.seq_name} ({e})")
                event_input = self.events
        else:
            event_input = self.events

        del events

        event_input = event_input.astype(np.float64)
        event_input_torch = torch.from_numpy(event_input)
        # if self.opt['phase'] != 'train':
        #     visualize_event_image(event_denoising,260,346,show=True)

        #-----------load img ---------------------
        gt_path = self.normal_img_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.low_img_paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']

            img_lq, img_gt, event = paired_random_crop_with_eventflow(img_lq,img_gt,
                                                                                              event_input_torch,
                                                                                              gt_size, scale,
                                                                                              gt_path)

            event_lq = self._generate_voxel_grid(event, height=img_lq.shape[0],width=img_lq.shape[1])
            if self.geometric_augs:

                img_lq, img_gt,event_lq = random_augmentation_with_voxel(img_lq, img_gt,event_lq)

        else:
            event_lq = self._generate_voxel_grid(event_input_torch, height=img_lq.shape[0],width=img_lq.shape[1])

            #event_lq  =event_input_torch

        img_gt, img_lq= img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'event': event_lq,
            'lq_path': lq_path,
            'gt_path': gt_path
        }







def pad_to_factor(H: int, W: int, factor: int):
    """返回需要补到右/下侧的 (pad_h, pad_w)，使 H/W 可被 factor 整除。"""
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    return pad_h, pad_w

def apply_reflect_pad_chw(x_chw: torch.Tensor, pad_h: int, pad_w: int):
    """对 CHW 张量仅在右/下侧做 reflect padding。"""
    if pad_h == 0 and pad_w == 0:
        return x_chw
    # F.pad 顺序: (left, right, top, bottom)
    return F.pad(x_chw, (0, pad_w, 0, pad_h), mode='reflect')

