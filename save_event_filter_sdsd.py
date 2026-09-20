import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
"""
可视化功能区
"""
def unified_spatial_grad(voxel):
    # voxel: (B, C, H, W)  -> aggregate over C
    voxel_mean = voxel.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           device=voxel.device, dtype=voxel.dtype).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           device=voxel.device, dtype=voxel.dtype).view(1,1,3,3)

    grad_x = F.conv2d(voxel_mean, sobel_x, padding=1)
    grad_y = F.conv2d(voxel_mean, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return grad_mag

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
#生成累积的事件图像（空间投影图）
# def visualize_event_image(voxel_grid, height, width,show_img=True,save_img=False,save_path=None):
#     """功能1：生成累积的事件图像（空间投影图）"""
#     if np.issubdtype(voxel_grid.dtype, np.floating):
#         dtype = voxel_grid.dtype  # 保持 float32/float64
#     else:
#         dtype = voxel_grid.dtype  # 保持 int 类型 (int32 等)
#         # 若想始终安全，也可不分支，统一用 float32:
#         # dtype = np.float32
#
#     full_frame = np.zeros((height, width), dtype=dtype)
#
#     for t_bin in range(voxel_grid.shape[0]):
#         full_frame += voxel_grid[t_bin]
#
#
#
#     plt.figure(figsize=(5, 5))
#     plt.imshow(full_frame, cmap='seismic', vmin=-2, vmax=2)
#     plt.title('Accumulated Event Frame')
#     plt.axis('off')
#     #
#     # if save_img:
#     #         if save_path is None:
#     #             save_path = "event_image.png"
#     #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
#     #         print(f"Image saved to {save_path}")
#     #
#     # plt.show()
#     # plt.imshow(full_frame, cmap='seismic', vmin=-2, vmax=2)
#     # plt.axis('off')  # 不显示坐标轴或边框
#
#     if save_img:
#         if save_path is None:
#             save_path = "event_image.png"
#         # ✅ 保存原始图像内容（和显示一致）
#         plt.imsave(save_path, full_frame, cmap='seismic', vmin=-2, vmax=2)
#         print(f"Image saved to {save_path}")
#     #if show_img:
#     #plt.show()
# 可视化3D体素网格
def visualize_voxel_grid(voxel_grid,angle_XTY):
    """功能2：体素网格的3D静态可视化（t,x, y）"""
    num_bins, height, width = voxel_grid.shape
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    for t_bin in range(num_bins):
        mask = voxel_grid[t_bin] != 0
        ys, xs = np.where(mask)
        ts = np.full_like(xs, t_bin)
        polarity_values = voxel_grid[t_bin, ys, xs] #获取每个时间点的事件极性
        colors = np.where(polarity_values > 0, 'red', 'blue')
        if angle_XTY:
            ax.scatter(xs, ts, ys, c=colors, s=2)
        else:
            ax.scatter(ts, xs, ys, c=colors, s=2)

    if angle_XTY:

        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('Y')
        #
        ax.set_xlim(0, width)
        ax.set_ylim(0, num_bins + 1)
        ax.set_zlim(0, height)#在这个值后面设置反转
        ax.invert_zaxis()
        ax.set_title('Voxel Grid (x, t, y)')
    else:
        ax.set_xlabel('T')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
       # ax.invert_zaxis()
        ax.set_xlim(0, num_bins + 1)
        ax.set_ylim(0, width)
        ax.set_zlim(0, height)#
        ax.invert_zaxis()
        ax.set_title('Voxel Grid (t, x, y)')
    plt.tight_layout()
    plt.show()


#可视化体素轨迹视图
def visualize_event_trajectory(voxel_grid):
    """功能3：体素轨迹视图（交换x<->t后观察运动趋势）"""
    num_bins, height, width = voxel_grid.shape
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')  # X <-> T 视角
    ax2 = fig.add_subplot(132)  # 累积图像
    ax3 = fig.add_subplot(133, projection='3d')  # 旋转视角

    xs_all, ys_all, ts_all, cols_all = [], [], [], []

    for t_bin in range(num_bins):
        mask = voxel_grid[t_bin] != 0
        ys, xs = np.where(mask)
        ts = np.full_like(xs, t_bin)
        polarity_values = voxel_grid[t_bin, ys, xs]
        colors = np.where(polarity_values > 0, 'red', 'blue')

        xs_all.append(xs)
        ys_all.append(ys)
        ts_all.append(ts)
        cols_all.extend(colors)

    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    ts_all = np.concatenate(ts_all)

    # 左图（x <-> t）
    ax1.scatter(ts_all, ys_all, xs_all, c=cols_all, s=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('X')
    ax1.set_xlim(0, num_bins - 1)
    ax1.set_ylim(0, height)
    ax1.set_zlim(0, width)
    ax1.set_title('Trajectory View (X <-> T)')

    # 中图（累积图）
    full_frame = np.zeros((height, width), dtype=np.int32)
    for t_bin in range(num_bins):
        full_frame += voxel_grid[t_bin]
    ax2.imshow(full_frame, cmap='seismic', vmin=-2, vmax=2)
    ax2.set_title('Accumulated Event Frame')
    ax2.axis('off')

    # 右图（旋转视角）
    ax3.scatter(xs_all, ys_all, ts_all, c=cols_all, s=2)
    ax3.view_init(elev=20, azim=30)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Time')
    ax3.set_xlim(0, width)
    ax3.set_ylim(0, height)
    ax3.set_zlim(0, num_bins - 1)
    ax3.set_title('Trajectory Rotated View')

    plt.tight_layout()
    plt.show()

#事件流可视化
def event_3dcloud(data,time_scale,angle_XTY):

    x, y, t_cont, p = get_continuous_event_coordinates(data, time_scale)

    colors = ['red' if pol > 0 else 'blue' for pol in p]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if angle_XTY :
        ax.scatter(x, t_cont, y, c=colors, s=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Continuous Time')
        ax.set_zlabel('Y')
        ax.invert_zaxis()
        ax.set_title('Continuous Event Stream in 3D')
    else:
        ax.scatter(t_cont, x, y, c=colors, s=2)
        ax.set_xlabel('Continuous Time')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.invert_zaxis()
        ax.set_title('Continuous Event Stream in 3D')
    #ax.scatter(x, y, t_cont, c=colors, s=2)


    plt.tight_layout()
    plt.show()

#展示RGB图像
def visual_img(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title("Image from HDF5")
    plt.axis('off')
    plt.show()
"""数据加载和处理区"""
#通用接口，加载h5或者npy文件
def load_event_data(file_path: str) -> np.ndarray:
    """
    通用接口：根据文件格式自动读取事件数据（支持 .npy 或 .h5）

    参数:
        file_path (str): 输入文件路径

    返回:
        data (np.ndarray): shape=(N, 4)，列顺序为 [x, y, timestamp, polarity]（极性为 ±1）
    """
    if file_path.endswith('.npy'):
        # 直接加载 .npy 文件
        data = np.load(file_path)
        if data.shape[1] != 4:
            raise ValueError("Invalid .npy shape, expected (N, 4)")
        return data

    elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
        # 加载 .h5 文件
        with h5py.File(file_path, 'r') as f:
            if 'events/x' in f:
                x = f['events/x'][:]
                y = f['events/y'][:]
                t = f['events/ts'][:]
                p = f['events/p'][:]
            elif 'events/xs' in f:
                x = f['events/xs'][:]
                y = f['events/ys'][:]
                t = f['events/ts'][:]
                p = f['events/ps'][:]
            else:
                raise ValueError("Unrecognized event format in .h5 file")

            polarity = np.where(p > 0, 1, -1)
            data = np.stack([x, y, t, polarity], axis=1)
            return data
    else:
        raise ValueError("Unsupported file format: expected .npy or .h5")

# 从事件数据生成体素网格（按照bins）
def generate_voxel_grid(data: np.ndarray, num_bins: int = 16):
    """
    从事件数据生成体素网格 (voxel grid)。

    参数:
        data (np.ndarray): 输入事件数组，形状为 (N, 4)，列依次为 [x, y, timestamp, polarity]
        num_bins (int): 时间轴划分的段数（默认 5）

    返回:
        voxel_grid (np.ndarray): 输出体素网格，形状为 (num_bins, H, W)
        height (int): 图像高度
        width (int): 图像宽度
    """
    if len(data) == 0 or data.shape[0] == 0:
        print("Warning: Empty event data, returning zero voxel grid")
        return np.zeros((num_bins, 260, 346), dtype=np.int32)
    t = data[:, 0].astype(np.int32)
    x= data[:, 1].astype(np.int32)
    y = data[:, 2]
    p = data[:, 3]

    #height = y.max() + 1
    #width = x.max() + 1

    # 时间归一化并量化为 bin 索引
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
    t_bins = (t_norm * (num_bins - 1)).astype(np.int32)

    # 初始化体素网格
    voxel_grid = np.zeros((num_bins, 260, 346), dtype=np.int32)

    # 填充体素网格
    for i in range(len(x)):
        polarity = 1 if p[i] > 0 else -1
        voxel_grid[t_bins[i], y[i], x[i]] += polarity

    return voxel_grid#, height, width

# 构造连续的事件流点云,即不将t归类至bins中；与event_3dcloud结合使用可查看连续事件流
def get_continuous_event_coordinates(data: np.ndarray, time_scale: float = 1.0):
    """
    提取连续时间坐标的事件点 (x, y, t_continuous)，用于连续轨迹可视化。

    参数:
        data (np.ndarray): 输入事件数组，形状为 (N, 4)，列为 [x, y, timestamp, polarity]
        time_scale (float): 时间缩放因子（默认归一化到 [0,1]；设为 num_bins 可放大到 bin 范围）

    返回:
        xs, ys, ts_cont, polarities: 对应事件的空间位置、连续时间、极性
    """
    x = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    t = data[:, 2]
    p = np.where(data[:, 3] > 0, 1, -1)

    # 将时间归一化到 [0, time_scale]
    t_min, t_max = t.min(), t.max()
    t_cont = ((t - t_min) / (t_max - t_min + 1e-6)) * time_scale

    return x, y, t_cont, p


"""复合功能函数"""
# 若一个h5中存有 多帧图像和对应event的时候
def process_h5_event_sequence(file_path, num_bins=5,angle_XTY=False,visualize_event_images=False,img_visualize=False,visual_3dcloud=False,visualize_voxel=False,save_img=False,path=None):
    with h5py.File(file_path, 'r') as f: #加载完整事件流
        print(list(f.keys()))
        xs = f['events/xs'][:]
        ys = f['events/ys'][:]
        ts = f['events/ts'][:]
        ps = f['events/ps'][:]
        file_name = os.path.splitext(os.path.basename(file_path))[0]#os.path.basename(file_path)提取路径末尾文件名（含扩展名)
        #os.path.splitext(...)[0]：去掉扩展名。
        # 获取图像组名，并按数字排序
        image_keys = sorted(
            list(f['images'].keys()),
            key=lambda name: int(name.replace("image", ""))
        )

        print(f"Total frames found: {len(image_keys)}")

        if save_img:
            dir = './save_img/'+file_name
            os.makedirs(dir, exist_ok=True)

        for i in range(len(image_keys) - 1):  # 每两个 image 之间形成一个段
            key0 = image_keys[i]
            #key1 = image_keys[i + 1]
            image_info =f['images'][key0] ## 此处取到的是一个具体的图像
            start_idx, end_idx = f['event_indices'][i]

            if img_visualize:  #可视化对应图像帧
                visual_img(image_info)

            # 截取对应事件段
            x = xs[start_idx:end_idx]
            y = ys[start_idx:end_idx]
            t = ts[start_idx:end_idx]
            p = ps[start_idx:end_idx]
            polarity = np.where(p > 0, 1, -1)

            data = np.stack([x, y, t, polarity], axis=1)

            # 可视化点云流
            if visual_3dcloud:
                event_3dcloud(data,time_scale=num_bins,angle_XTY=angle_XTY)

            # 标准化到每个bins的数据
            voxel_grid, height, width = generate_voxel_grid(data, num_bins=num_bins)

            if visualize_voxel: #可视化离散事件
                visualize_voxel_grid(voxel_grid,angle_XTY) #当bins=1是可将视角切换来观察所有的离散event被堆叠到同一个bins后和下面的图像的区别是什么
            if visualize_event_images: #可视化对应图像
                if save_img:
                    path = os.path.join(dir, f"{key0}_event_frame.png")
                visualize_event_image(voxel_grid, height, width,save_img=save_img,save_path=path)

def process_npy_event_sequence(event_file, index_file, num_bins=5, angle_XTY=False, visualize_event_images=False, img_visualize=False, visual_3dcloud=False, visualize_voxel=False, save_img=False, save_path=None):
    """
       Process event sequences from .npy files, visualizing or saving event images.

       :param event_file: path to the .npy file containing event data (x, y, t, p).
       :param index_file: path to the .npy file containing event indices corresponding to frames.
       :param num_bins: number of bins for voxelization.
       :param angle_XTY: whether to visualize 3D cloud in X-T-Y angle.
       :param visualize_event_images: whether to visualize event images.
       :param img_visualize: whether to visualize the images.
       :param visual_3dcloud: whether to visualize 3D point clouds.
       :param visualize_voxel: whether to visualize the voxel grid.
       :param save_img: whether to save images.
       :param save_path: directory to save the images (if save_img is True).
       """
    # Load event data from the .npy file
    events = np.load(event_file)  # Shape: (num_events, 4), columns: x, y, t, p
    xs, ys, ts, ps = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    # Load event indices for each frame from the index file
    event_indices = np.load(index_file)  # Shape: (num_frames, 2), each row: [start_idx, end_idx]

    # Extract the file name without extension for saving
    file_name = os.path.splitext(os.path.basename(event_file))[0]

    # Directory to save images if needed
    if save_img:
        dir = os.path.join(save_path, file_name)
        os.makedirs(dir, exist_ok=True)

    for i in range(len(event_indices) - 1):  # Iterate over event frames (between image frames)
        start_idx, end_idx = event_indices[i]

        # Extract corresponding events for the current frame
        x = xs[start_idx:end_idx]
        y = ys[start_idx:end_idx]
        t = ts[start_idx:end_idx]
        p = ps[start_idx:end_idx]
        polarity = np.where(p > 0, 1, -1)

        data = np.stack([x, y, t, polarity], axis=1)

        # Visualization: 3D cloud of events
        if visual_3dcloud:
            event_3dcloud(data, time_scale=num_bins, angle_XTY=angle_XTY)

        # Voxel grid generation and visualization
        voxel_grid, height, width = generate_voxel_grid(data, num_bins=num_bins)

        if visualize_voxel:  # Visualize voxel grid
            visualize_voxel_grid(voxel_grid, angle_XTY)

        # Visualize event images and save if required
        if visualize_event_images:
            if save_img:
                image_path = os.path.join(dir, f"event_frame_{i}.png")
            visualize_event_image(voxel_grid, height, width, save_img=save_img, save_path=image_path)

    print(f"Processing completed for {file_name}.")


def visualize_events(evs, H, W):
    """
    可视化事件流为图像，事件的空间坐标 (x, y) 和极性 (p) 用不同颜色表示。
    :param evs: [N, 4] 事件数据，其中每一行是 [x, y, t, p]
    :param H: 图像的高度
    :param W: 图像的宽度
    :return: 可视化后的图像
    """
    # 提取事件的 x, y, t, p
    xs, ys, ts, ps = evs[:, 0], evs[:, 1], evs[:, 2], evs[:, 3]
    x = xs.astype(np.int64)
    y = ys.astype(np.int64)
    pol = ps
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0

    # 创建空白图像，背景填充为白色
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')

    # 将极性转换为 -1 和 1
    pol = pol.astype('int')
    pol[pol == 0] = -1

    # 直接在事件的坐标位置上操作，不做裁剪
    img[y, x] = [255, 0, 0]  # 负极性事件 → 红色
    img[y[pol == 1], x[pol == 1]] = [0, 0, 255]  # 正极性事件 → 蓝色

    # 显示图像
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title('Event Visualization')
    plt.axis('off')
    plt.show()

    return img


def eq24_eq25(events: torch.Tensor, S: torch.Tensor, omega: float = 0.02):
    """
    实现 Zhang et al. (TIP 2024) 论文中公式(24)-(25) 的操作。

    Args:
        events: Tensor, (bins, H, W)，事件体素
        S:      Tensor, (3, H, W)，RGB图像 [0,1]
        omega:  float, 梯度监督水平 ω

    Returns:
        E_dot:  Tensor, (bins, H, W)，筛选后的事件体素
        g_mask: Tensor, (H, W)，梯度掩码
    """
    # ---- (1) 将图像转灰度 ----
    gray = S.mean(dim=0, keepdim=True)  # (1,H,W)

    # ---- (2) 计算梯度 (∇xS) ----
    # 使用简单的有限差分代替 Sobel
    dx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1))  # (1,H,W)
    dy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1))  # (1,H,W)
    grad_mag = torch.sqrt(dx ** 2 + dy ** 2)  # (1,H,W)

    # ---- (3) 找到 q = ∇xS 中出现次数最多的像素值 (用中值近似) ----
    q = grad_mag.median()

    # ---- (4) 根据(25)计算 g_j ----
    mask_keep = (grad_mag < (q - omega)) | (grad_mag > (q + omega))
    g_mask = torch.where(mask_keep, grad_mag, torch.zeros_like(grad_mag))  # (1,H,W)

    # ---- (5) 根据(24) 对事件进行筛选 ----
    # 积分项 \int δ(t-t_i) dt = 1，所以仅需根据 g_mask≠0 筛选
    E_dot = events * (g_mask > 0).float()  # (bins,H,W)

    return E_dot#, g_mask.squeeze(0)

def adaptive_threshold_grad_mask(grad_mag, omega=0.02, win_size=5):  #自适应q
    """
    根据局部窗口计算自适应阈值 q(x,y)，得到保留掩码。
    grad_mag : (1,H,W)
    omega     : 区分阈值范围
    win_size  : 局部窗口大小
    """
    kernel = torch.ones(1, 1, win_size, win_size, device=grad_mag.device) / (win_size**2)
    # 局部平均作为局部 q
    local_mean = F.conv2d(grad_mag.unsqueeze(0), kernel, padding=win_size//2)
    # 自适应保留
    mask_keep = (grad_mag < (local_mean.squeeze(0) - omega)) | (grad_mag > (local_mean.squeeze(0) + omega))
    g_mask = torch.where(mask_keep, grad_mag, torch.zeros_like(grad_mag))
    return g_mask

def temporal_consistency_filter(voxel_orig, voxel_denoised, window=3, thresh=0.05):
    """
    以原事件作为基础，用去噪结果判断时间一致性，以防信息被误删
    """
    # 对去噪体素做时间平滑
    x = voxel_denoised.unsqueeze(0).unsqueeze(0)  # (1,1,b,H,W)
    kernel = torch.ones(1,1,window,1,1,device=voxel_orig.device)/window
    smooth = F.conv3d(x, kernel, padding=(window//2,0,0))

    # 在时间窗口内若平均响应高，认为时间上稳定
    mask_temporal = (smooth.squeeze(0).squeeze(0).abs() > thresh).float()

    # 用这个掩码过滤原始事件体素
    voxel_filtered = voxel_orig * mask_temporal

    return voxel_filtered

def eq24_eq25_improved(events, S, omega=0.02):
    import torch.nn.functional as F

    gray = S.mean(dim=0, keepdim=True)
    # 1) 对数亮度灰度化
    #gray = torch.log(S.mean(dim=0, keepdim=True) + 1e-6)

    # 2) 计算梯度
    dx = F.pad(gray[:, :, 1:] - gray[:, :, :-1], (0, 1))
    dy = F.pad(gray[:, 1:, :] - gray[:, :-1, :], (0, 0, 0, 1))
    grad_mag = torch.sqrt(dx**2 + dy**2)

    # #全局阈值策略
    # q = grad_mag.median()
    # mask_keep = (grad_mag < (q - omega)) | (grad_mag > (q + omega))
    # g_mask = torch.where(mask_keep, grad_mag, torch.zeros_like(grad_mag))

    # 3) 局部自适应阈值
    g_mask = adaptive_threshold_grad_mask(grad_mag, omega=omega, win_size=5)

    # 4) 梯度掩码作用在事件上
    E_dot = events * (g_mask > 0).float()


    return E_dot

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
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L

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

if  __name__ == '__main__':
    import os
    from pathlib import Path

    # 根目录
    root_list = ['/mnt/datasets/Lowlight/SDE/SDE_in','/mnt/datasets/Lowlight/SDE/SDE_out']
    for root_path in root_list:
        root_path = Path(root_path)
        events_folder = 'events'
        png_folder = 'low'#low_img
        output_folder = 'events_denoising'
        omega = 0.01
        visual_folder = f'denoising_visual_low_{omega}'
        visual_low_folder = 'denoising_visual_event'
        # ==== 收集所有场景文件夹下的文件 ====
        file_list = []
        png_list = []

        # 遍历 root_path 下的所有场景文件夹
        scene_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])

        for scene_folder in scene_folders:
            # 构建当前场景的 events 和 low 路径
            scene_events_folder = scene_folder / events_folder
            scene_low_folder = scene_folder / png_folder

            # 检查文件夹是否存在
            if scene_events_folder.exists() and scene_low_folder.exists():
                # 收集当前场景的事件文件和图像
                scene_event_files = sorted([
                    str(f) for f in scene_events_folder.iterdir()
                    if f.suffix == ".npz" and "lowlight_event" not in f.name
                ])
                scene_png_files = sorted([
                    str(f) for f in scene_low_folder.iterdir()
                    if f.suffix == ".png"
                ])

                file_list.extend(scene_event_files)
                png_list.extend(scene_png_files)

        # 全局排序
        file_list = sorted(file_list)
        png_list = sorted(png_list)

        print(f"Found {len(file_list)} event files and {len(png_list)} png files")

        # 加载模型
        Decom_net = Decom().cuda()
        Decom_net = load_initialize(Decom_net, '/mnt/yzs/code/Retinexformer_Bilevel/ckpt/init_low.pth')

        # 处理文件
        for path, img_path in zip(file_list, png_list):
            ev = np.load(path)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()

            with torch.no_grad():
                R_tensor, L_tensor = Decom_net(img_tensor)
            R_tensor = R_tensor.squeeze(0).cpu()

            # 获取文件名
            path_name = os.path.splitext(os.path.basename(path))[0]
            print(f"Processing file: {path_name}")

            ev = ev["arr_0"] if "arr_0" in ev else ev

            voxel = generate_voxel_grid(ev)

            voxel_denosing = eq24_eq25_improved(torch.from_numpy(voxel).float(), R_tensor, omega=omega)
            voxel_denoised = voxel_denosing.cpu().numpy()



            # 构建保存路径（保持场景文件夹结构）
            event_file_path = Path(path)
            scene_name = event_file_path.parent.parent.name  # 获取场景文件夹名称

            # 构建输出路径
            output_scene_folder = root_path / scene_name / output_folder
            output_visual_folder = root_path / scene_name / visual_folder
            output_visual_low_folder = root_path / scene_name / visual_low_folder
            # 创建输出文件夹（如果不存在）
            if not output_scene_folder.exists():
                os.makedirs(output_scene_folder)  # 使用 makedirs 而不是 mkdir，可以创建多级目录
                print(f"Created output folder: {output_scene_folder}")
            if not output_visual_folder.exists():
                os.makedirs(output_visual_folder)  # 使用 makedirs 而不是 mkdir，可以创建多级目录
                print(f"Created output folder: {output_visual_folder}")
            if not output_visual_low_folder.exists():
                os.makedirs(output_visual_low_folder)  # 使用 makedirs 而不是 mkdir，可以创建多级目录
                print(f"Created output folder: {output_visual_low_folder}")

            save_visual_path = output_visual_folder / f'{path_name}.png'
            save_visual_low_path =output_visual_low_folder /f'{path_name}.png'
            save_path = output_scene_folder / f'{path_name}.npz'
            # visualize_event_image(voxel_denoised, 260, 346, saveimg=True,savepath=save_visual_path)
            # visualize_event_image(torch.from_numpy(voxel).float(),260,346,saveimg=True,savepath=save_visual_low_path)
            # 保存文件
            np.savez(save_path, arr_0=voxel_denoised)
            print(f"Saved to: {save_path}")
