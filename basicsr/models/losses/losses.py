import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision
from basicsr.models.losses.loss_util import weighted_loss
from basicsr.models.losses.ssim_utils import *
from basicsr.models.losses.ssim_utils import _ssim
import  lpips
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss  # 把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss  # 把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction


#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss

# ---------------- SSIM Loss ---------------- #
class SSIMLoss(nn.Module):
    """
    与 L1CharbonnierLoss 风格一致的构造方式
    loss = 1 - SSIM
    """

    def __init__(self, window_size=11, size_average=True, weight=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.weight = weight

    def forward(self, pred, target):
        # 支持 [C,H,W] 或 [B,C,H,W] 输入
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        B, C, H, W = pred.shape

        window = create_window(self.window_size, C).to(pred.device)
        ssim_val = _ssim(pred, target, window, self.window_size, C, self.size_average)
        return (1.0 - ssim_val) * self.weight


class StructAwareTVLoss(nn.Module):
    """
    结构感知 TV（对 I 做平滑），R/重建图只作为引导，不回传梯度。
    用法与 SSIMLoss / L1Loss 一致：forward(pred, target)，其中
      pred   = I（照明，B×(1或3)×H×W）
      target = guide（引导，通常传 R 或 R*I）
    可在 YAML 里通过 pred_key/target_key 绑定到 I 与 guide。
    """

    def __init__(self,
                 lambda_edge: float = 6.0,
                 use_sobel: bool = False,
                 detach_guide: bool = True,
                 weight: float = 1.0):
        super().__init__()
        self.lambda_edge = float(lambda_edge)
        self.use_sobel = bool(use_sobel)
        self.detach_guide = bool(detach_guide)
        self.weight = float(weight)

        # 构建梯度核：2x2 差分（简洁）或 3x3 Sobel（更抗噪）
        if not self.use_sobel:
            kx = torch.tensor([[0., 0.],
                               [-1., 1.]]).view(1, 1, 2, 2)
            ky = kx.transpose(2, 3)
        else:
            kx = torch.tensor([[1., 0., -1.],
                               [2., 0., -2.],
                               [1., 0., -1.]]).view(1, 1, 3, 3)
            ky = kx.transpose(2, 3)

        # 注册为 buffer，自动随模型搬到正确 device/dtype
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        # x: (B,1/3,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.size(1) == 1:
            return x
        # RGB -> gray
        return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def _conv_grad(self, x: torch.Tensor, dir_: str) -> torch.Tensor:
        k = self.kx if dir_ == 'x' else self.ky
        # reflect/replicate 都可；这里用 replicate，避免边界伪梯度
        pad = (k.shape[-1] // 2, k.shape[-1] // 2, k.shape[-2] // 2, k.shape[-2] // 2)
        x = F.pad(x, pad, mode='replicate')
        # 自动广播到 dtype/device
        k = k.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, k, stride=1, padding=0)

    def _grad_abs(self, x: torch.Tensor, dir_: str) -> torch.Tensor:
        return self._conv_grad(x, dir_).abs()

    def _avg_grad(self, x: torch.Tensor, dir_: str) -> torch.Tensor:
        g = self._grad_abs(x, dir_)
        return F.avg_pool2d(g, kernel_size=3, stride=1, padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : I（B×(1或3)×H×W）
        target : guide（R 或 R*I，B×(1或3)×H×W）
        返回： weight * mean( |∇I| * exp(-λ * avg(|∇guide|)) ) 逐方向相加
        """
        # 形状兼容：支持 [C,H,W]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        Ig = self._to_gray(pred)  # I 灰度
        Gg = self._to_gray(target)  # 引导灰度
        if self.detach_guide:
            Gg = Gg.detach()

        # I 的梯度
        gx_I = self._grad_abs(Ig, 'x')
        gy_I = self._grad_abs(Ig, 'y')

        # 引导的平滑边缘强度 -> 权重
        wx = torch.exp(-self.lambda_edge * self._avg_grad(Gg, 'x'))
        wy = torch.exp(-self.lambda_edge * self._avg_grad(Gg, 'y'))

        loss = (wx * gx_I).mean() + (wy * gy_I).mean()
        return loss * self.weight


class TVLoss(nn.Module):
    """Total Variation (TV) loss.

    Encourages spatial smoothness by penalizing large gradients in the output.

    Args:
        loss_weight (float): Weight for TV loss. Default: 1.0.
        reduction (str): Specifies the reduction mode: 'mean' | 'sum' | 'none'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TVLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: 'none' | 'mean' | 'sum'."
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    @staticmethod
    def unified_spatial_grad(voxel: torch.Tensor):
        """Compute spatial gradient magnitude using Sobel operators."""
        voxel_mean = voxel.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=voxel.device, dtype=voxel.dtype
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=voxel.device, dtype=voxel.dtype
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(voxel_mean, sobel_x, padding=1)
        grad_y = F.conv2d(voxel_mean, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad_mag, grad_x, grad_y

    def forward(self, pred: torch.Tensor, target=None, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W). The predicted denoised output.
            target (Tensor, optional): Not used here (for API compatibility).
            weight (Tensor, optional): Optional element-wise weight.
        """
        _, grad_x, grad_y = self.unified_spatial_grad(pred)
        loss = torch.abs(grad_x) + torch.abs(grad_y)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class WeightedBCELoss(nn.Module):
    """
    自动根据 target 计算 pos_weight 的版本
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # 自动计算正负样本比例
        num_pos = target.sum()
        num_neg = target.numel() - num_pos
        pos_weight = num_neg / (num_pos + 1e-6)

        # 创建损失函数并计算
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=pred.device),
            reduction=self.reduction
        )
        loss = loss_fn(pred, target)
        return loss * self.weight


class WeightedCrossEntropyLoss(nn.Module):
    """
    自动根据 target 计算各类别权重的版本（适用于多分类任务，比如事件相机的三分类）
    定义结构保持与 WeightedBCELoss 一致
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, C, BINS, H, W]  —— 网络输出的 logits
            target: [B, BINS, H, W]     —— 类别标签 (0..C-1)
        """
        # 如果输入是 [B, BINS, H, W] 这样 3维（缺 batch），自动加一维（保持你原来逻辑）
        if pred.dim() == 4:  # [C, BINS, H, W]
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        B, C, BINS, H, W = pred.shape

        # ========= 自动计算每类出现次数 =========
        with torch.no_grad():
            # 展平 target 统计每个类别像素数
            freq = torch.bincount(target.flatten(), minlength=C).float() + 1e-6
            # 类别权重反比出现频次（出现少的类权重大）
            weights = 1.0 / freq
            # 归一化，使权重和为类别数 C
            weights = weights / weights.sum() * C
            weights = weights.to(pred.device)

        # ========= 定义带动态权重的 CrossEntropy =========
        loss_fn = nn.CrossEntropyLoss(weight=weights, reduction=self.reduction)

        # ========= 计算损失 =========
        loss = loss_fn(pred, target)

        return loss * self.weight
# class GradientSmoothLoss(nn.Module):
#     """Gradient smoothness regularization loss.
#     Penalizes non-smooth changes in spatial gradients of E_pred.
#
#     Args:
#         loss_weight (float): weight for this regularization.
#         reduction (str): reserved for API consistency.
#     Usage:
#         loss = GradientSmoothLoss(loss_weight=1e-4)
#         penalty = loss(E_pred)
#     """
#
#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(GradientSmoothLoss, self).__init__()
#         self.loss_weight = float(loss_weight)
#         self.reduction = reduction
#
#     def forward(self, pred ):
#         """Args:
#             pred (Tensor): model output, shape (N, C, H, W)
#             target: unused, kept for compatibility
#         """
#         _, grad_x, grad_y = unified_spatial_grad(pred)
#         loss = gradient_smooth_loss_from_outputs(grad_x, grad_y)
#         return self.loss_weight * loss
#
# def gradient_smooth_loss_from_outputs(gx, gy):
#     gx_dx, gx_dy = gx[:, :, :, :-1] - gx[:, :, :, 1:], gx[:, :, :-1, :] - gx[:, :, 1:, :]
#     gy_dx, gy_dy = gy[:, :, :, :-1] - gy[:, :, :, 1:], gy[:, :, :-1, :] - gy[:, :, 1:, :]
#     return (gx_dx.abs().mean() + gx_dy.abs().mean() + gy_dx.abs().mean() + gy_dy.abs().mean())
#
# def unified_spatial_grad(voxel):
#     # voxel: (B, C, H, W)  -> aggregate over C
#     voxel_mean = voxel.mean(dim=1, keepdim=True)
#     sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
#                            device=voxel.device, dtype=voxel.dtype).view(1,1,3,3)
#     sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
#                            device=voxel.device, dtype=voxel.dtype).view(1,1,3,3)
#
#     grad_x = F.conv2d(voxel_mean, sobel_x, padding=1)
#     grad_y = F.conv2d(voxel_mean, sobel_y, padding=1)
#     grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
#     return grad_mag,grad_x,grad_y

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l1', reduction='mean'):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss


# ---------------------------------------------------------------
# define the edge loss to enhance the deblurring task
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l2', reduction='mean'):
        super(EdgeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).cuda()

        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.criterion(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight
