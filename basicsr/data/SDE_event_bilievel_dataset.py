

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
        self.event_de_folder = root / opt['events_denoising_subdir']
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
                    ev_denoise_paths = self.event_de_folder / event_name
                    if not ev_denoise_paths.exists():
                        raise FileNotFoundError(
                            f"Event file for low light image {event_name} not found in {self.event_folder}")
                    self.ev_denoise_paths.append(str(ev_denoise_paths))
                    self.ev_list.append(ev_denoise_paths.name)


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
            events_denoising_path = self.ev_denoise_paths[index]
            if ev_path is None:
                raise FileNotFoundError(f"No split event npz for low frame: {self.low_img_list[index]}")
            events = np.load(ev_path)
            events_denoising = np.load(events_denoising_path)
        else:
            raise ValueError('w/o assign event')

        try:
            self.events = events["arr_0"] if "arr_0" in events else events  #[t,x,y,p]
            self.event_denoising = events_denoising["arr_0"]# if "arr_0" in events_denoising else events_denoising
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
            event_denoising = self.event_denoising

        del events

        event_input = event_input.astype(np.float64)
        event_denoising = event_denoising.astype(np.float64)
        event_input_torch = torch.from_numpy(event_input)
        event_denoising = torch.from_numpy(event_denoising)
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

            img_lq, img_gt, event,event_denoising  = paired_random_crop_with_eventflow(img_lq,img_gt,
                                                                                              event_input_torch,
                                                                                              event_denoising,
                                                                                              gt_size, scale,
                                                                                              gt_path)

            event_lq = self._generate_voxel_grid(event, height=img_lq.shape[0],width=img_lq.shape[1])
            event_denoising = self._generate_voxel_grid(event_denoising, height=img_lq.shape[0],width=img_lq.shape[1])
            if self.geometric_augs:

                img_lq, img_gt,event_lq,event_denoising = random_augmentation_with_voxel(img_lq, img_gt,event_lq,event_denoising)

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
            'event_denoising': event_denoising,
            'lq_path': lq_path,
            'gt_path': gt_path
        }




# 正常的体素划分法
class Dataset_EventFlow_SDE(data.Dataset):
    def __init__(self, opt,
        events_subdir='events',
        low_subdir='low',
        normal_subdir='normal',
        # ==== 保留 whole-event 模式的路径（可不传）====
        whole_low_event_npz=None,
        whole_normal_event_npz=None,
                 ):
        super(Dataset_EventFlow_SDE, self).__init__()
        self.dataset_root = opt.dataset_root
        # ---- 基本参数 ----
        self.H = opt.height
        self.W = opt.width
        self.is_train = opt.is_train
        # ==== CHANGED: 无论训练/验证，都初始化裁剪尺寸 ====
        self.center_cropped_height = opt.height
        self.random_cropped_width = opt.width

        self.seq_name = "merged"  # ==== CHANGED: 统一标识 ====
        self.voxel_grid_channel = opt.voxel_grid_channel
        self.is_split_event = opt.is_split_event
        root = Path(self.dataset_root)
        self.low_img_folder = str(root / low_subdir)  # ==== CHANGED: 仅为兼容原字段名 ====
        self.noraml_img_folder = str(root / normal_subdir)  # ==== CHANGED: 兼容原字段名 ====
        low_dir = root / low_subdir
        normal_dir = root / normal_subdir
        events_dir = root / events_subdir if events_subdir else None

        # ---- 1) 全量读取 + 排序 + 按索引配对（不要求同名） ----
        low_pngs = sorted([p for p in low_dir.iterdir() if p.suffix.lower() == '.png'])
        normal_pngs = sorted([p for p in normal_dir.iterdir() if p.suffix.lower() == '.png'])

        n = min(len(low_pngs), len(normal_pngs))
        if n == 0:
            raise RuntimeError(f'No png files under {low_dir} or {normal_dir}')
        if len(low_pngs) != len(normal_pngs):
            print(f"[warn] low({len(low_pngs)}) != normal({len(normal_pngs)}); pairing first {n} by index")

            # ==== CHANGED: 直接保存完整路径与文件名 ====
        self.low_img_paths = [str(p) for p in low_pngs[:n]]
        self.normal_img_paths = [str(p) for p in normal_pngs[:n]]
        self.low_img_list = [p.name for p in low_pngs[:n]]
        self.normal_img_list = [p.name for p in normal_pngs[:n]]

        # ---- 2) 事件路径：按 low 同名 .npz，优先 events/，否则回退 low/ ----
        if self.is_split_event:
            self.ev_paths = []
            self.ev_normal_paths = []

            for lp, np in zip(low_pngs[:n], normal_pngs[:n]):
                npz_name = lp.with_suffix('.npz').name
                npz_name_normal = np.with_suffix('.npz').name

                ev_path = None
                ev_normal_path = None

                if events_dir and events_dir.exists():
                    cand = events_dir / npz_name
                    if cand.exists():
                        ev_path = cand
                if ev_path is None:
                    cand = low_dir / npz_name
                    if cand.exists():
                        ev_path = cand

                # 查找 normal 的事件文件
                if events_dir and events_dir.exists():
                    cand_normal = events_dir / npz_name_normal
                    if cand_normal.exists():
                        ev_normal_path = cand_normal

                if ev_normal_path is None:
                    cand_normal = normal_dir / npz_name_normal
                    if cand_normal.exists():
                        ev_normal_path = cand_normal

                self.ev_paths.append(str(ev_path) if ev_path is not None else None)
                self.ev_normal_paths.append(str(ev_normal_path) if ev_normal_path is not None else None)
        else:
            # whole-event 模式（合并后通常不建议用；保留兼容）
            self.low_event_file = (
                str(root / "lowlight_event.npz") if whole_low_event_npz is None else whole_low_event_npz
            )
            self.normal_event_file = (
                str(root / "normalight_event.npz") if whole_normal_event_npz is None else whole_normal_event_npz
            )

        self.num_input = n
        self.ev_idx = None
        self.events = None

    def __len__(self):
        return self.num_input
    #从这还没改
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
        if self.is_train:
            width, height = self.random_cropped_width, self.center_cropped_height
        event_start = event[0, 0]
        event_end = event[-1, 0]

        ch = (event[:, 0].to(torch.float32) / (event_end - event_start) * self.voxel_grid_channel).long()
        torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1

        voxel_grid = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)
        return voxel_grid

    def __getitem__(self, index):
        # 1) event
        if (self.events is None) and (self.is_split_event == False):
            events = np.load(self.low_event_file)
            events_normal = np.load(self.normal_event_file)
        elif self.is_split_event == True:
            ev_path = self.ev_paths[index]
            ev_normal_path = self.ev_normal_paths[index]
            if ev_path is None:
                raise FileNotFoundError(f"No split event npz for low frame: {self.low_img_list[index]}")
            if ev_normal_path is None:
                raise FileNotFoundError(f"No split event npz for normal: {self.normal_img_list[index]}")
            events = np.load(ev_path)
            events_normal = np.load(ev_normal_path)
        else:
            raise ValueError('w/o assign event')

        try:
            self.events = events["arr_0"] if "arr_0" in events else events
            self.events_normal = events_normal["arr_0"] if "arr_0" in events_normal else events_normal
            if self.events.ndim == 1:  # 结构化数组
                et = self.events["timestamp"]
                ex = self.events["x"]
                ey = self.events["y"]
                ep = self.events["polarity"]
                self.events = np.stack([et, ex, ey, ep], axis=1)
                et_normal = self.events_normal["timestamp"]
                ex_normal = self.events_normal["x"]
                ey_normal = self.events_normal["y"]
                ep_normal = self.events_normal["polarity"]
                self.events_normal = np.stack([et_normal, ex_normal, ey_normal, ep_normal], axis=1)
        except Exception as e:
            print(f"loading event error @ index: {index} ({e})")

        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
            except Exception as e:
                print(f"loading event error @ seq: {self.seq_name} ({e})")
                event_input = self.events
                event_gt = self.events_normal
        else:
            event_input = self.events
            event_gt   = self.events_normal

        del events
        del events_normal
        # 2) images & illumination
        img_low = cv2.cvtColor(cv2.imread(self.low_img_paths[index]), cv2.COLOR_BGR2RGB)
        img_blur = cv2.blur(img_low, (5, 5))
        img_low_illumination_map = self._illumiantion_map(img_low)

        img_gt = cv2.cvtColor(cv2.imread(self.normal_img_paths[index]), cv2.COLOR_BGR2RGB)

        img_low_illumination_map = np.expand_dims(img_low_illumination_map / 255.0, axis=-1)

        event_input_torch = torch.from_numpy(event_input)
        event_gt_torch = torch.from_numpy(event_gt)

        if self.is_train:
            crop_img_list, crop_event_list = self._crop(
                [img_low, img_gt, img_low_illumination_map, img_blur],
                [event_input_torch,event_gt_torch],
            )
            input_voxel_grid_list = []
            for crop_event in crop_event_list:
                crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
                input_voxel_grid_list.append(self._generate_voxel_grid(crop_event))
            pad_hw = (0, 0)
        else:

            crop_npy_list, crop_event_list = [img_low, img_gt, img_low_illumination_map, img_blur], [event_input_torch,event_gt_torch]
            crop_img_list = [torch.from_numpy(arr).permute(2, 0, 1).float() for arr in crop_npy_list]

            # === 计算并应用镜像 padding，仅在右/下侧 ===
            # 备注：如果你知道 U-Net 的总下采样层数为 level，factor=2**level；不确定时多数网络是 16 或 32
            factor = getattr(self, "pad_factor", 16)  # 你也可以在 __init__ 里设置 self.pad_factor
            H, W = crop_img_list[0].shape[-2:]
            pad_h, pad_w = pad_to_factor(H, W, factor)

            if pad_h or pad_w:
                crop_img_list[0] = apply_reflect_pad_chw(crop_img_list[0], pad_h, pad_w)  # low
                # crop_img_list[1] = apply_reflect_pad_chw(crop_img_list[1], pad_h, pad_w)  # gt
                crop_img_list[2] = apply_reflect_pad_chw(crop_img_list[2], pad_h, pad_w)  # illum (1,H,W)
                crop_img_list[3] = apply_reflect_pad_chw(crop_img_list[3], pad_h, pad_w)
            pad_hw = (H, W)

            # 事件体素：直接按 padded 尺寸生成（相当于 0-padding）
            H_pad, W_pad = H + pad_h, W + pad_w
            input_voxel_grid_list = []
            for crop_event in crop_event_list:
                input_voxel_grid = self._generate_voxel_grid(crop_event, H_pad, W_pad)
                input_voxel_grid_list.append(input_voxel_grid)

        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )

        sample = {
            "lowligt_image": crop_img_list[0] / 255.0,
            "normalligt_image": crop_img_list[1] / 255.0,
            "event_free": input_voxel_grid_list[0],
            "event_normal": input_voxel_grid_list[1],
            "lowlight_image_blur": crop_img_list[3] / 255.0,
            "ill_list": [crop_img_list[2]],
            "seq_name": self.seq_name,
            "frame_id": Path(self.low_img_paths[index]).stem,
            "pad_hw": pad_hw if not self.is_train else (0, 0),  # 仅测试时有效
        }

        # reduce memory cost
        self.events = None
        return sample




#计数图法
class Dataset_EventFlow_cnt_SDE(data.Dataset):
    def __init__(self, opt,
                 events_subdir='events',
                 low_subdir='low',
                 normal_subdir='normal',
                 grad_subdir='grad',
                 event_normal_subdir='events_normal',
                 # ==== 保留 whole-event 模式的路径（可不传）====
                 whole_low_event_npz=None,
                 whole_normal_event_npz=None,
                 ):
        super(Dataset_EventFlow_cnt_SDE, self).__init__()
        self.dataset_root = opt['dataset_root']
        # ---- 基本参数 ----
        self.H = opt['height']
        self.W = opt['width']
        self.geometric_augs = opt.get("geometric_augs", False)

        self.opt = opt
        self.is_train = opt["is_train"]
        # ==== CHANGED: 无论训练/验证，都初始化裁剪尺寸 ====
        self.center_cropped_height = opt['height']
        self.random_cropped_width = opt['width']

        self.seq_name = "merged"  # ==== CHANGED: 统一标识 ====
        self.voxel_grid_channel = opt["voxel_grid_channel"]
        self.is_split_event =True
        root = Path(self.dataset_root)
        self.low_img_folder = str(root / low_subdir)  # ==== CHANGED: 仅为兼容原字段名 ====
        self.noraml_img_folder = str(root / normal_subdir)  # ==== CHANGED: 兼容原字段名 ====
        grad_dir = root / grad_subdir
        low_dir = root / low_subdir
        normal_dir = root / normal_subdir
        events_dir = root / events_subdir if events_subdir else None
        events_normal_dir = root / event_normal_subdir

        # ---- 1) 全量读取 + 排序 + 按索引配对（不要求同名） ----
        low_pngs = sorted([p for p in low_dir.iterdir() if p.suffix.lower() == '.png'])
        normal_pngs = sorted([p for p in normal_dir.iterdir() if p.suffix.lower() == '.png'])
        grad_pngs = sorted([p for p in grad_dir.iterdir() if p.suffix.lower() == '.png'])
        n = min(len(low_pngs), len(normal_pngs))
        if n == 0:
            raise RuntimeError(f'No png files under {low_dir} or {normal_dir}')
        if len(low_pngs) != len(normal_pngs):
            print(f"[warn] low({len(low_pngs)}) != normal({len(normal_pngs)}); pairing first {n} by index")

            # ==== CHANGED: 直接保存完整路径与文件名 ====
        self.low_img_paths = [str(p) for p in low_pngs[:n]]
        self.normal_img_paths = [str(p) for p in normal_pngs[:n]]
        self.grad_img_paths = [str(p) for p in grad_pngs[:n]]
        self.low_img_list = [p.name for p in low_pngs[:n]]
        self.normal_img_list = [p.name for p in normal_pngs[:n]]
        self.grad_img_list = [p.name for p in grad_pngs[:n]]

        # ---- 2) 事件路径：按 low 同名 .npz，优先 events/，否则回退 low/ ----
        if self.is_split_event:
            self.ev_paths = []
            self.ev_normal_paths = []

            for lp, np in zip(low_pngs[:n], normal_pngs[:n]):
                npz_name = lp.with_suffix('.npz').name
                npz_name_normal = np.with_suffix('.npy').name

                ev_path = None
                ev_normal_path = None

                if events_dir and events_dir.exists():
                    cand = events_dir / npz_name
                    if cand.exists():
                        ev_path = cand
                if ev_path is None:
                    cand = low_dir / npz_name
                    if cand.exists():
                        ev_path = cand

                    # 查找 normal 的事件文件
                if events_normal_dir and events_normal_dir.exists():
                    cand_normal = events_normal_dir / npz_name_normal
                    if cand_normal.exists():
                        ev_normal_path = cand_normal

                if ev_normal_path is None:
                    cand_normal = events_normal_dir / npz_name_normal
                    if cand_normal.exists():
                        ev_normal_path = cand_normal

                self.ev_paths.append(str(ev_path) if ev_path is not None else None)
                self.ev_normal_paths.append(str(ev_normal_path) if ev_normal_path is not None else None)
        else:
            # whole-event 模式（合并后通常不建议用；保留兼容）
            self.low_event_file = (
                str(root / "lowlight_event.npz") if whole_low_event_npz is None else whole_low_event_npz
            )
            self.normal_event_file = (
                str(root / "normalight_event.npz") if whole_normal_event_npz is None else whole_normal_event_npz
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
        if self.is_train:
            width, height = self.random_cropped_width, self.center_cropped_height
        event_start = event[0, 0]
        event_end = event[-1, 0]

        ch = (event[:, 0].to(torch.float32) / (event_end - event_start) * self.voxel_grid_channel).long()
        torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1

        voxel_grid = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)
        return voxel_grid


    def create_cnt_encoding(self, events, sensor_resolution):
        """
        events: torch.tensor, 4xN [x, y, t, p]

        return: count: torch.tensor, 2 x H x W
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        return events_to_channels(xs, ys, ps, sensor_size=sensor_resolution)

    def __getitem__(self, index):
        # 1) event
        if (self.events is None) and (self.is_split_event == False):
            events = np.load(self.low_event_file)
            events_normal = np.load(self.normal_event_file)
        elif self.is_split_event == True:
            ev_path = self.ev_paths[index]
            ev_normal_path = self.ev_normal_paths[index]
            if ev_path is None:
                raise FileNotFoundError(f"No split event npz for low frame: {self.low_img_list[index]}")
            if ev_normal_path is None:
                raise FileNotFoundError(f"No split event npz for normal: {self.normal_img_list[index]}")
            events = np.load(ev_path)
            events_normal = np.load(ev_normal_path)
        else:
            raise ValueError('w/o assign event')

        try:
            self.events = events["arr_0"] if "arr_0" in events else events
            #self.events_normal = events_normal["arr_0"] if "arr_0" in events_normal else events_normal
            if self.events.ndim == 1:  # 结构化数组
                et = self.events["timestamp"]
                ex = self.events["x"]
                ey = self.events["y"]
                ep = self.events["polarity"]
                self.events = np.stack([et, ex, ey, ep], axis=1)
                # et_normal = self.events_normal["timestamp"]
                # ex_normal = self.events_normal["x"]
                # ey_normal = self.events_normal["y"]
                # ep_normal = self.events_normal["polarity"]
                #self.events_normal = np.stack([et_normal, ex_normal, ey_normal, ep_normal], axis=1)
            self.events_normal = events_normal
        except Exception as e:
            print(f"loading event error @ index: {index} ({e})")

        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
                event_input = self.events
                event_gt = self.events_normal
            except Exception as e:
                print(f"loading event error @ seq: {self.seq_name} ({e})")
        else:
                event_input = self.events
                event_gt = self.events_normal


        del events
        del events_normal
        # 2) images & illumination
        img_low = cv2.cvtColor(cv2.imread(self.low_img_paths[index]), cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(cv2.imread(self.normal_img_paths[index]), cv2.COLOR_BGR2RGB)
        img_grad = cv2.cvtColor(cv2.imread(self.grad_img_paths[index]),cv2.COLOR_BGR2GRAY)

        event_input_torch = torch.from_numpy(event_input)
        event_gt_torch = torch.from_numpy(event_gt)
        gt_size = self.opt["gt_size"]
        if self.is_train:

            #crop
            img_lq, img_grad, img_gt, event_input_torch,event_gt_torch = paired_random_crop_with_eventflow(img_low, img_grad,
                                                                                                  img_gt,
                                                                                                 event_input_torch,event_gt_torch, gt_size
                                                                                                 )
            event_vox = self._generate_voxel_grid(event_input_torch, height=gt_size, width=gt_size)
            event_gt_vox = self._generate_voxel_grid(event_gt_torch, height=gt_size, width=gt_size)

            visualize_event_image(event_vox,260,346)
            visualize_event_image(event_gt_vox, 260, 346)
            if self.geometric_augs:
                img_gt, img_lq, img_grad, event_vox,event_gt_vox= random_augmentation_with_Flow(
                    img_lq, img_grad, img_gt, event_vox,event_gt_vox,sensor_size=(self.center_cropped_height, self.random_cropped_width)
                )
            visualize_event_image(event_vox, 260, 346)
            visualize_event_image(event_gt_vox, 260, 346)

            event_cnt = self.create_cnt_encoding(event_input_torch,
                                                 [self.center_cropped_height, self.random_cropped_width])
            event_gt_cnt = self.create_cnt_encoding(event_gt_torch,
                                                    [self.center_cropped_height, self.random_cropped_width])
            pad_hw = (0, 0)

        else:

           # crop_npy_list, crop_event_list = [img_low, img_gt], [event_input_torch, event_gt_torch]
            img_low = torch.from_numpy(img_low).permute(2, 0, 1).float()
            img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).float()
            img_grad = torch.from_numpy(img_grad).float()#.permute(2, 0, 1).float()

            # === 计算并应用镜像 padding，仅在右/下侧 ===
            # 备注：如果你知道 U-Net 的总下采样层数为 level，factor=2**level；不确定时多数网络是 16 或 32
            factor = getattr(self, "pad_factor", 16)  # 你也可以在 __init__ 里设置 self.pad_factor
            H, W = img_low.shape[-2:]
            pad_h, pad_w = pad_to_factor(H, W, factor)

            if pad_h or pad_w:
                img_low = apply_reflect_pad_chw(img_low, pad_h, pad_w)  # low
                # crop_img_list[1] = apply_reflect_pad_chw(crop_img_list[1], pad_h, pad_w)  # gt
                img_gt = apply_reflect_pad_chw(img_gt, pad_h, pad_w)  # illum (1,H,W)
                img_grad = apply_reflect_pad_chw(img_grad, pad_h, pad_w)
            pad_hw = (H, W)

            # 事件体素：直接按 padded 尺寸生成（相当于 0-padding）
            H_pad, W_pad = H + pad_h, W + pad_w


            event_vox = self._generate_voxel_grid(event_input_torch, H_pad, W_pad)
            event_gt_vox = self._generate_voxel_grid(event_gt_torch, H_pad, W_pad)



            event_cnt = self.create_cnt_encoding(event_vox,[H_pad, W_pad])
            event_gt_cnt = self.create_cnt_encoding(event_gt_vox,[H_pad,W_pad])

        del (
            event_input,
            event_input_torch,
        )

        sample = {
            "lowligt_image":img_low / 255.0,
            "normalligt_image": img_gt / 255.0,
            "event_free": event_vox,
            "event_cnt": event_cnt,
            "event_normal": event_gt_vox,
            "event_normal_cnt": event_gt_cnt,
            "seq_name": self.seq_name,
            "frame_id": Path(self.low_img_paths[index]).stem,
            "pad_hw": pad_hw if not self.is_train else (0, 0),  # 仅测试时有效
        }

        # reduce memory cost
        self.events = None
        return sample




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

class Dataset_PairedImage_Slide(data.Dataset):
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
        super(Dataset_PairedImage_Slide, self).__init__()
        self.opt = opt
        # file client (io backend) 文件客户端
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        # self.img_num = len(self.paths)

        h, w = 400, 600  # img shape
        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        print('patches per line is: %d' % (self.patch_per_line))
        print('patches per colum is: %d' % (self.patch_per_colum))
        print('The number of images is: %d' % (len(self.paths)))
        print('The number of patches is: %d' % ((len(self.paths)) * self.patch_per_img))

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[::-1, :, :].copy()
        return img

    def __getitem__(self, index):
        # 把index当做patch的序列号，先定位到image的序列号，然后根据(h_idx,w_idx)读图
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # scale = self.opt['scale']
        # index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        stride = self.opt['stride']
        crop_size = self.opt['gt_size']
        img_idx, patch_idx = index // self.patch_per_img, index % self.patch_per_img  # 这里的indx是指总共patch的序号，
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line

        img_idx = img_idx % len(self.paths)

        # data loading
        gt_path = self.paths[img_idx]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[img_idx]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # print(img_lq.shape)
        img_lq = img_lq[h_idx * stride: h_idx * stride + crop_size, w_idx * stride: w_idx * stride + crop_size, :]
        img_gt = img_gt[h_idx * stride: h_idx * stride + crop_size, w_idx * stride: w_idx * stride + crop_size, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            img_lq = self.arguement(img_lq, rotTimes, vFlip, hFlip)
            img_gt = self.arguement(img_gt, rotTimes, vFlip, hFlip)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([np.ascontiguousarray(img_gt), np.ascontiguousarray(img_lq)],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths) * self.patch_per_img


class Dataset_PairedImage_Norm(data.Dataset):
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
        super(Dataset_PairedImage_Norm, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        # img_gt = (img_gt - img_gt.min())/(img_gt.max()-img_gt.min())
        img_lq = (img_lq - img_lq.min()) / (img_lq.max() - img_lq.min())

        # stx()
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)
        # stx()
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_GaussianDenoising(data.Dataset):
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
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)

            # flip, rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                              bgr2rgb=True,
                                              float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

