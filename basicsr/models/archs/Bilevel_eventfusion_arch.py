import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
import numpy as np

#from tensorflow.python.ops.numpy_ops import ones_like
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
import os

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
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]

class EIAB(nn.Module):   #(Event2Img Attention Block)
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                EI_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim,FeedForward(dim=dim))
            ]))
    def forward(self, x, event):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """

        x = x.permute(0, 2, 3, 1)
        event = event.permute(0, 2, 3, 1)

        for (attn, ff) in self.blocks:
            x = attn(x, event_fea=event) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class EI_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
    def forward(self,x_in,event_fea):
        b, h, w, c = x_in.shape
        n = h * w

        img_flat = x_in.reshape(b, n, c)  # [b, n, c_img]
        event_flat = event_fea.reshape(b, n, c)  # [b, n, c_event]

        q_inp = self.to_q(img_flat)  # [b, n, heads*dim_head]
        k_inp = self.to_k(event_flat)  # [b, n, heads*dim_head]
        v_inp = self.to_v(event_flat)# [b, n, heads*dim_head]

        q = rearrange(q_inp, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k_inp, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v_inp, 'b n (h d) -> b h n d', h=self.num_heads)

        q = q.transpose(-2, -1)  # [b, h, d, n]
        k = k.transpose(-2, -1)  # [b, h, d, n]
        v = v.transpose(-2, -1)  # [b, h, d, n]

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1))  # [b, h, d, d]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v  # [b, h, d, n]
        x = x.permute(0, 3, 1, 2)  # [b, n, h, d]
        x = x.reshape(b, n, self.num_heads * self.dim_head)  # [b, n, c]
        out_c = self.proj(x).view(b, h, w, c)  # [b, h, w, c]

        # 深度可分离卷积的位置编码
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)  # [b, h, w, c]

        out = out_c + out_p
        return out


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class IG_MSA(nn.Module): #此处做了照明分支可选操作，若不传入照明分支，则为正常的注意力操作
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in, illu_fea_trans=None):
        """
        x_in:            [b, h, w, c]
        illu_fea_trans:  [b, h, w, c] 或 None
                         - None: baseline（不使用照明增益）
        return:          [b, h, w, c]
        """
        b, h, w, c = x_in.shape
        n = h * w

        # 展平空间维
        x = x_in.reshape(b, n, c)              # [b, n, c]
        q_inp = self.to_q(x)                   # [b, n, heads*dim_head]
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        # 分头
        q = rearrange(q_inp, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k_inp, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v_inp, 'b n (h d) -> b h n d', h=self.num_heads)

        # ====== 照明门控（可选）======
        if illu_fea_trans is not None:
            # 期望 illu_fea_trans 的通道数与 dim 对齐：c = heads*dim_head
            # （通常 IGAB 构造时已保证 dim_level = heads * dim_head）
            illu = illu_fea_trans.flatten(1, 2)                      # [b, n, c]
            illu = rearrange(illu, 'b n (h d) -> b h n d', h=self.num_heads)  # [b, h, n, d]
        else:
            # baseline：不改变 v，相当于全 1 门控
            illu = torch.ones_like(v)

        v = v * illu
        # ============================

        # 标准注意力
        q = q.transpose(-2, -1)     # [b, h, d, n]
        k = k.transpose(-2, -1)     # [b, h, d, n]
        v = v.transpose(-2, -1)     # [b, h, d, n]

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1))       # [b, h, d, d]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v                            # [b, h, d, n]
        x = x.permute(0, 3, 1, 2)               # [b, n, h, d]
        x = x.reshape(b, n, self.num_heads * self.dim_head)  # [b, n, c]
        out_c = self.proj(x).view(b, h, w, c)   # [b, h, w, c]

        # 深度可分离卷积的位置编码
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)                   # [b, h, w, c]

        out = out_c + out_p
        return out



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea=None):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            if illu_fea :
                x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x   #原有照度分支
            else:
                x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x) #扩充通道

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class Denoiser_nolu_E(nn.Module):  #去除了光照分支
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser_nolu_E, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder（仅保留特征分支：IGAB + 特征下采样）
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i],
                     dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
                               heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder（结构不变：上采样 + 1x1 融合 + IGAB）
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2,
                                   stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                     dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function（可留作备用）
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x:          [b,c,h,w]   （仅特征，无照明分支）
        return out: [b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)

        # Encoder（仅特征分支）
        fea_encoder = []
        for (igab, FeaDownSample) in self.encoder_layers:
            fea = igab(fea)              # b,c,h,w
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = LeWinBlock(fea)

        # Mapping（残差输出）
        out = self.mapping(fea) #+ x
        #out = torch.sigmoid(out)
        return out


# class Denoiser_fusion(nn.Module):
#
#     def __init__(self, in_dim=3, out_dim=3, voxel_dim=16, dim=31, level=2, num_blocks=[2, 4, 4]):
#         super(Denoiser_fusion, self).__init__()
#         self.dim = dim
#         self.level = level
#
#         self.embedding_r = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
#         self.embedding_e = nn.Conv2d(voxel_dim, self.dim, 3, 1, 1, bias=False)
#
#         self.encoder_r = nn.ModuleList([])
#         self.encoder_e = nn.ModuleList([])
#
#         dim_level = dim
#
#         for i in range(level):
#             self.encoder_r.append(nn.ModuleList([
#                 IGAB(dim=dim_level, num_blocks=num_blocks[i],
#                      dim_head=dim, heads=dim_level // dim),
#                 nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
#             ]))
#             self.encoder_e.append(nn.ModuleList([
#                 IGAB(dim=dim_level, num_blocks=num_blocks[i],
#                      dim_head=dim, heads=dim_level // dim),
#                 nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
#             ]))
#             dim_level *= 2
#
#         self.bottleneck_r = IGAB(dim=dim_level, dim_head=dim,
#                                  heads=dim_level // dim, num_blocks=num_blocks[-1])
#         self.bottleneck_e = IGAB(dim=dim_level, dim_head=dim,
#                                  heads=dim_level // dim, num_blocks=num_blocks[-1])
#
#         # 只保留事件到图像的特征融合（EIAB）
#         self.feature_fusion_r = EIAB(dim=dim_level,
#                                      dim_head=dim,
#                                      heads=dim_level // dim,
#                                      num_blocks=2,
#                                      )
#
#
#         self.decoder_r = nn.ModuleList([])
#         self.decoder_e = nn.ModuleList([])
#         self.decoder_fusion_r = nn.ModuleList([])
#
#         for i in range(level):
#             self.decoder_r.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_level, dim_level // 2,
#                                    stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
#                 nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
#                 IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
#                      dim_head=dim, heads=(dim_level // 2) // dim),
#             ]))
#
#             # 只保留事件到图像的融合
#             self.decoder_fusion_r.append(nn.ModuleList([
#                 EIAB(dim=dim_level // 2, num_blocks=2,
#                      dim_head=dim, heads=(dim_level // 2) // dim)
#             ]))
#
#             self.decoder_e.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_level, dim_level // 2,
#                                    stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
#                 nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
#                 IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
#                      dim_head=dim, heads=(dim_level // 2) // dim),
#             ]))
#             dim_level //= 2
#
#         self.mapping_r = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
#         self.mapping_e = nn.Conv2d(self.dim, 3*voxel_dim, 3, 1, 1, bias=False)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x, event):
#         input = x
#
#         # Embedding
#         fea_x = self.embedding_r(input)
#         fea_e = self.embedding_e(event)
#
#         # Encoder
#         fea_encoder_r = []
#         fea_encoder_e = []
#
#         for (igab_r, down_r), (igab_e, down_e) in zip(self.encoder_r, self.encoder_e):
#             fea_x = igab_r(fea_x)  # b,c,h,w
#             fea_encoder_r.append(fea_x)
#             fea_x = down_r(fea_x)
#
#             fea_e = igab_e(fea_e)
#             fea_encoder_e.append(fea_e)
#             fea_e = down_e(fea_e)
#
#         # Bottleneck
#         fea_x = self.bottleneck_r(fea_x)
#         fea_e = self.bottleneck_e(fea_e)
#
#         # 只进行事件到图像的特征融合（单向）
#         fea_x = self.feature_fusion_r(fea_x, fea_e)
#
#         # Decoder
#         for i, (dec_r, dec_e, fusion_block_r) in enumerate(zip(self.decoder_r, self.decoder_e, self.decoder_fusion_r)):
#             # R分支（图像）
#             up_r, fusion_r, igab_r = dec_r
#             up_e, fusion_e, igab_e = dec_e
#             fusion_block = fusion_block_r[0]
#
#             fea_x = up_r(fea_x)
#             fea_e = up_e(fea_e)
#
#             fea_x = fusion_r(
#                 torch.cat([fea_x, fea_encoder_r[self.level - 1 - i]], dim=1)
#             )
#             fea_e = fusion_e(
#                 torch.cat([fea_e, fea_encoder_e[self.level - 1 - i]], dim=1)
#             )
#
#             # 只进行事件到图像的特征融合（单向）
#             fea_x = fusion_block(fea_x, fea_e)
#
#             fea_x = igab_r(fea_x)
#             fea_e = igab_e(fea_e)
#
#         out_r = self.mapping_r(fea_x) + x
#         out_r = torch.sigmoid(out_r)
#         out_e = self.mapping_e(fea_e)
#
#         return out_r, out_e


# class Denoiser_fusion(nn.Module):
#
#     def __init__(self, in_dim=3, out_dim=3, voxel_dim=16, dim=31, level=2, num_blocks=[2, 4, 4]):
#         super(Denoiser_fusion, self).__init__()
#         self.dim = dim
#         self.level = level
#
#         self.embedding_r = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
#         self.embedding_e = nn.Conv2d(voxel_dim, self.dim, 3, 1, 1, bias=False)
#         # 用于增强事件的二次嵌入
#         self.embedding_e_enhanced = nn.Conv2d(3*voxel_dim, self.dim, 3, 1, 1, bias=False)
#
#         self.encoder_r = nn.ModuleList([])
#         self.encoder_e = nn.ModuleList([])
#
#         dim_level = dim
#
#         for i in range(level):
#             self.encoder_r.append(nn.ModuleList([
#                 IGAB(dim=dim_level, num_blocks=num_blocks[i],
#                      dim_head=dim, heads=dim_level // dim),
#                 nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
#             ]))
#             self.encoder_e.append(nn.ModuleList([
#                 IGAB(dim=dim_level, num_blocks=num_blocks[i],
#                      dim_head=dim, heads=dim_level // dim),
#                 nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
#             ]))
#             dim_level *= 2
#
#         self.bottleneck_r = IGAB(dim=dim_level, dim_head=dim,
#                                  heads=dim_level // dim, num_blocks=num_blocks[-1])
#         self.bottleneck_e = IGAB(dim=dim_level, dim_head=dim,
#                                  heads=dim_level // dim, num_blocks=num_blocks[-1])
#
#         # 事件到图像的特征融合（EIAB）
#         self.feature_fusion_r = EIAB(dim=dim_level,
#                                      dim_head=dim,
#                                      heads=dim_level // dim,
#                                      num_blocks=2,
#                                      )
#
#         self.decoder_r = nn.ModuleList([])
#         self.decoder_e = nn.ModuleList([])
#         self.decoder_fusion_r = nn.ModuleList([])
#
#         for i in range(level):
#             self.decoder_r.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_level, dim_level // 2,
#                                    stride=2, kernel_size=2, padding=0, output_padding=0),
#                 nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
#                 IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
#                      dim_head=dim, heads=(dim_level // 2) // dim),
#             ]))
#
#             self.decoder_fusion_r.append(nn.ModuleList([
#                 EIAB(dim=dim_level // 2, num_blocks=2,
#                      dim_head=dim, heads=(dim_level // 2) // dim)
#             ]))
#
#             self.decoder_e.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_level, dim_level // 2,
#                                    stride=2, kernel_size=2, padding=0, output_padding=0),
#                 nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
#                 IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
#                      dim_head=dim, heads=(dim_level // 2) // dim),
#             ]))
#             dim_level //= 2
#
#         self.mapping_r = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
#         self.mapping_e = nn.Conv2d(self.dim, 3*voxel_dim, 3, 1, 1, bias=False)  # 修改输出通道
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x, event):
#         """
#         两阶段处理:
#         Stage 1: 事件分支预测增强事件
#         Stage 2: 使用增强事件辅助图像增强
#         """
#
#         # ==================== Stage 1: 事件分支处理 ====================
#         # Embedding
#         fea_e = self.embedding_e(event)
#
#         # Encoder
#         fea_encoder_e = []
#         for igab_e, down_e in self.encoder_e:
#             fea_e = igab_e(fea_e)
#             fea_encoder_e.append(fea_e)
#             fea_e = down_e(fea_e)
#
#         # Bottleneck
#         fea_e = self.bottleneck_e(fea_e)
#
#         # Decoder
#         for i, dec_e in enumerate(self.decoder_e):
#             up_e, fusion_e, igab_e = dec_e
#             fea_e = up_e(fea_e)
#             fea_e = fusion_e(
#                 torch.cat([fea_e, fea_encoder_e[self.level - 1 - i]], dim=1)
#             )
#             fea_e = igab_e(fea_e)
#
#         # 事件分支输出 - 增强后的事件
#         out_event = self.mapping_e(fea_e)
#
#
#         # ==================== Stage 2: 图像分支处理（使用增强事件） ====================
#         # 使用增强后的事件重新嵌入
#         fea_e_enhanced = self.embedding_e_enhanced(out_event)
#
#         # 图像嵌入
#         fea_x = self.embedding_r(x)
#
#         # Encoder（图像和增强事件）
#         fea_encoder_r = []
#         fea_encoder_e_enhanced = []
#
#         for (igab_r, down_r), (igab_e, down_e) in zip(self.encoder_r, self.encoder_e):
#             fea_x = igab_r(fea_x)
#             fea_encoder_r.append(fea_x)
#             fea_x = down_r(fea_x)
#
#             fea_e_enhanced = igab_e(fea_e_enhanced)
#             fea_encoder_e_enhanced.append(fea_e_enhanced)
#             fea_e_enhanced = down_e(fea_e_enhanced)
#
#         # Bottleneck
#         fea_x = self.bottleneck_r(fea_x)
#         fea_e_enhanced = self.bottleneck_e(fea_e_enhanced)
#
#         # 事件到图像的特征融合
#         fea_x = self.feature_fusion_r(fea_x, fea_e_enhanced)
#
#         # Decoder（带融合）
#         for i, (dec_r, fusion_block_r) in enumerate(zip(self.decoder_r, self.decoder_fusion_r)):
#             up_r, fusion_r, igab_r = dec_r
#             up_e, fusion_e, igab_e = self.decoder_e[i]
#             fusion_block = fusion_block_r[0]
#
#             fea_x = up_r(fea_x)
#             fea_e_enhanced = up_e(fea_e_enhanced)
#
#             fea_x = fusion_r(
#                 torch.cat([fea_x, fea_encoder_r[self.level - 1 - i]], dim=1)
#             )
#             fea_e_enhanced = fusion_e(
#                 torch.cat([fea_e_enhanced, fea_encoder_e_enhanced[self.level - 1 - i]], dim=1)
#             )
#
#             # 事件到图像的特征融合
#             fea_x = fusion_block(fea_x, fea_e_enhanced)
#
#             fea_x = igab_r(fea_x)
#             fea_e_enhanced = igab_e(fea_e_enhanced)
#
#         # 图像分支输出
#         out_r = self.mapping_r(fea_x) + x
#         out_r = torch.sigmoid(out_r)
#
#         return out_r, out_event


class Denoiser_nolu(nn.Module):  #去除了光照分支
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser_nolu, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder（仅保留特征分支：IGAB + 特征下采样）
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i],
                     dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
                               heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder（结构不变：上采样 + 1x1 融合 + IGAB）
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2,
                                   stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                     dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function（可留作备用）
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x:          [b,c,h,w]   （仅特征，无照明分支）
        return out: [b,c,h,w]
        """
        # Embedding
        fea = self.embedding(x)

        # Encoder（仅特征分支）
        fea_encoder = []
        for (igab, FeaDownSample) in self.encoder_layers:
            fea = igab(fea)              # b,c,h,w
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = LeWinBlock(fea)

        # Mapping（残差输出）
        out = self.mapping(fea) #+ x
        out = torch.sigmoid(out)
        return out

# class Denoiser_nolu_R(nn.Module):  #去除了光照分支
#     def __init__(self, in_dim=3, out_dim=3,voxel_dim=16, dim=31, level=2, num_blocks=[2, 4, 4]):
#         super(Denoiser_nolu_R, self).__init__()
#         self.dim = dim
#         self.level = level
#
#         # Input projection
#         self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
#         #self.embedding_e = nn.Conv2d(3*voxel_dim,voxel_dim,3,1,1,bias=False)
#         #self.fusion = nn.Conv2d(voxel_dim+self.dim,self.dim,3,1,1,bias=False)
#         # Encoder（仅保留特征分支：IGAB + 特征下采样）
#         self.encoder_layers = nn.ModuleList([])
#         dim_level = dim
#         for i in range(level):
#             self.encoder_layers.append(nn.ModuleList([
#                 IGAB(dim=dim_level, num_blocks=num_blocks[i],
#                      dim_head=dim, heads=dim_level // dim),
#                 nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
#             ]))
#             dim_level *= 2
#
#         # Bottleneck
#         self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
#                                heads=dim_level // dim, num_blocks=num_blocks[-1])
#
#         # Decoder（结构不变：上采样 + 1x1 融合 + IGAB）
#         self.decoder_layers = nn.ModuleList([])
#         for i in range(level):
#             self.decoder_layers.append(nn.ModuleList([
#                 nn.ConvTranspose2d(dim_level, dim_level // 2,
#                                    stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
#                 nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
#                 IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
#                      dim_head=dim, heads=(dim_level // 2) // dim),
#             ]))
#             dim_level //= 2
#
#         # Output projection
#         self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
#
#         # activation function（可留作备用）
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x,events):
#         """
#         x:          [b,c,h,w]   （仅特征，无照明分支）
#         return out: [b,c,h,w]
#         """
#         # Embedding
#         fea = self.embedding(x)
#
#         # Encoder（仅特征分支）
#         fea_encoder = []
#         for (igab, FeaDownSample) in self.encoder_layers:
#             fea = igab(fea)              # b,c,h,w
#             fea_encoder.append(fea)
#             fea = FeaDownSample(fea)
#
#         # Bottleneck
#         fea = self.bottleneck(fea)
#
#         # Decoder
#         for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
#             fea = FeaUpSample(fea)
#             fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
#             fea = LeWinBlock(fea)
#
#         # Mapping（残差输出）
#         out = self.mapping(fea) + x
#         out = torch.sigmoid(out)
#         return out
#
#

# class EventFormer_Bilevel(nn.Module):
#     def __init__(self, in_channels=3, voxel_channel=16,out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1],mode='train',path=None,level=2):
#         super(EventFormer_Bilevel, self).__init__()
#         self.stage = stage
#         self.mode = mode
#
#         self.Lnet = Denoiser_nolu(in_dim=1, out_dim=1, dim=n_feat, level=level,
#                                       num_blocks=num_blocks)
#         #
#         # self.Rnet = Denoiser_nolu_R(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
#         #                               num_blocks=num_blocks)
#         self.REnet = Denoiser_fusion(in_dim=in_channels, out_dim=out_channels,voxel_dim = voxel_channel, dim=n_feat, level=level,
#                                       num_blocks=num_blocks)
#
#         self.Decom_low = Decom()
#         self.Decom_high = Decom()
#         self.model_Decom_low = load_initialize(self.Decom_low, path)
#         high_path  = path.replace("init_low", "init_high")
#         self.model_Decom_high = load_initialize(self.Decom_high, high_path)
#
#         # self.register_buffer(
#         #     'event_class_values',
#         #     torch.tensor([0.0, 1.0, -1.0]).view(1, 3, 1, 1, 1)
#         # )
#
#     def get_structured_event(self, event_logits, temperature=0.5):
#         """
#         将事件分类 logits 转换为结构化事件（可微）
#
#         Args:
#             event_logits: (B, 3, bins, H, W) - 分类 logits
#             temperature: Softmax 温度
#
#         Returns:
#             structured_event: (B, bins, H, W) - 值域 [0, 2]，可微
#         """
#         # Softmax with temperature
#         event_prob = F.softmax(event_logits / temperature, dim=1)
#         # (B, 3, bins, H, W)
#
#         # 加权求和：期望类别值
#         structured_event = (event_prob * self.event_class_values).sum(dim=1)
#         # (B, bins, H, W)
#
#         return structured_event
#     def forward(self, x,event,x_high=None):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         B, Bins, H, W = event.shape
#         input = x
#         if self.mode == 'train':
#             R_0,L_0 = self.model_Decom_low(input)
#             R_1,L_1 = self.model_Decom_high(x_high)
#         else:
#             R_0,L_0 = self.model_Decom_low(input)
#             # R_1,L_1 = torch.ones_like(L_0), torch.ones_like(R_0)
#             R_1, L_1 = self.model_Decom_high(x_high)
#
#         L = self.Lnet(L_0)
#         R,event = self.REnet(R_0,event)
#         event_logits = event.view(B, 3, Bins, H, W)
#         # event_prob = F.softmax(event_logits, dim=1) #软阈值
#         # structured_voxel = (event_prob * self.event_class_values).sum(dim=1)
#         #event = torch.argmax(event_logits, dim=1)  #硬阈值法
#
#         #R = self.Rnet(R_0,event)
#
#         out = L * R
#         #out = self.body(x)
#
#
#         return out,L,R,L_1,R_1,event_logits

class Denoiser_nolu_R(nn.Module):  #去除了光照分支
    def __init__(self, in_dim=3, out_dim=3,voxel_dim=16, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser_nolu_R, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.embedding_e = nn.Conv2d(voxel_dim, self.dim, 3, 1, 1, bias=False)

        # ✅ 融合层：将图像特征和事件特征融合
        self.fusion = nn.Conv2d(self.dim * 2, self.dim, 1, 1, bias=False)
        # Encoder（仅保留特征分支：IGAB + 特征下采样）


        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i],
                     dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # FeaDownSample
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim,
                               heads=dim_level // dim, num_blocks=num_blocks[-1])

        # ==================== 事件分支（轻量级编码器） ====================
        # ✅ 简化的事件编码器：只用Conv进行特征提取和下采样
        self.encoder_layers_e = nn.ModuleList([])
        dim_level_e = dim
        for i in range(level):
            self.encoder_layers_e.append(nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim_level_e, dim_level_e, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(dim_level_e),
                    nn.ReLU(inplace=True)
                ),
                nn.Conv2d(dim_level_e, dim_level_e * 2, 4, 2, 1, bias=False)
            ]))
            dim_level_e *= 2

        # ✅ 事件Bottleneck（简化版）
        self.bottleneck_e = nn.Sequential(
            nn.Conv2d(dim_level_e, dim_level_e, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim_level_e),
            nn.ReLU(inplace=True)
        )

        self.bottleneck_fusion = EIAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=2
        )

        # ==================== Decoder ====================
        self.decoder_layers = nn.ModuleList([])
        self.decoder_fusion_layers = nn.ModuleList([])

        # Decoder（结构不变：上采样 + 1x1 融合 + IGAB）
        self.decoder_layers = nn.ModuleList([])

        dim_level = dim * (2 ** level)

        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2,
                                   stride=2, kernel_size=2, padding=0, output_padding=0),  # FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # Fusion: concat 后通道还原
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i],
                     dim_head=dim, heads=(dim_level // 2) // dim),
            ]))

            self.decoder_fusion_layers.append(
                EIAB(
                    dim=dim_level // 2,
                    dim_head=dim,
                    heads=(dim_level // 2) // dim,
                    num_blocks=2
                )
            )

            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function（可留作备用）
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,events):
        """
        x:          [b,c,h,w]   （仅特征，无照明分支）
        return out: [b,c,h,w]
        """
        # Embedding
        fea_img  = self.embedding(x)
        fea_event = self.embedding_e(events)

        fea = torch.cat([fea_img, fea_event], dim=1)  # [b, dim*2, h, w]
        fea = self.fusion(fea)

        # Encoder（仅特征分支）
        fea_encoder = []
        fea_encoder_e = []
        for (igab, FeaDownSample) in self.encoder_layers:
            fea = igab(fea)              # b,c,h,w
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # ✅ 事件编码（轻量级，只用Conv）
        fea_e = fea_event
        for (conv_block, downsample) in self.encoder_layers_e:
            fea_e = conv_block(fea_e)
            fea_encoder_e.append(fea_e)  # 下采样前保存
            fea_e = downsample(fea_e)

        # Bottleneck
        fea = self.bottleneck(fea)

        fea_e = self.bottleneck_e(fea_e)

        # ✅ Bottleneck融合（Event to Image）
        fea = self.bottleneck_fusion(fea, fea_e)

        # Decoder
        for i, (decoder_layer, fusion_layer) in enumerate(
                zip(self.decoder_layers, self.decoder_fusion_layers)
        ):
            FeaUpSample, Fusion, LeWinBlock = decoder_layer

            # 上采样
            fea = FeaUpSample(fea)

            # Skip connection融合
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))

            # ✅ Event to Image融合（使用对应层级的事件特征）
            fea = fusion_layer(fea, fea_encoder_e[self.level - 1 - i])

            # IGAB处理
            fea = LeWinBlock(fea)

        # Mapping（残差输出）
        out = self.mapping(fea) + x
        out = torch.sigmoid(out)
        return out



class EventFormer_Bilevel_event_fusion(nn.Module):
    def __init__(self, in_channels=3, voxel_channel=16,out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1],mode='train',path=None,level=2):
        super(EventFormer_Bilevel_event_fusion, self).__init__()
        self.stage = stage
        self.mode = mode

        self.Lnet = Denoiser_nolu(in_dim=1, out_dim=1, dim=n_feat, level=level,
                                      num_blocks=num_blocks)

        self.Rnet = Denoiser_nolu_R(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                      num_blocks=num_blocks)
        self.Enet = Denoiser_nolu_E(in_dim=voxel_channel, out_dim=3*voxel_channel, dim=n_feat, level=level,
                                      num_blocks=num_blocks)
        self.Decom_low = Decom()
        self.Decom_high = Decom()
        self.model_Decom_low = load_initialize(self.Decom_low, path)
        high_path  = path.replace("init_low", "init_high")
        self.model_Decom_high = load_initialize(self.Decom_high, high_path)

        # ✅ 定义 1D 张量（用于索引）
        self.register_buffer(
            'event_class_values_1d',
            torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
        )

        # ✅ 定义 5D 张量（用于广播乘法）
        self.register_buffer(
            'event_class_values_5d',
            torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).view(1, 3, 1, 1, 1)
        )

    def forward(self, x,event,x_high=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        B, Bins, H, W = event.shape
        input = x
        if self.mode == 'train':
            R_0,L_0 = self.model_Decom_low(input)
            R_1,L_1 = self.model_Decom_high(x_high)
        else:
            R_0,L_0 = self.model_Decom_low(input)
            # R_1,L_1 = torch.ones_like(L_0), torch.ones_like(R_0)
            R_1, L_1 = self.model_Decom_high(x_high)

        L = self.Lnet(L_0)
        event = self.Enet(event)
        event_logits = event.view(B, 3, Bins, H, W)
        if self.mode == 'train':
            event_prob = F.softmax(event_logits, dim=1) #软阈值
            event = (event_prob * self.event_class_values_5d).sum(dim=1)
        else:
            event = torch.argmax(event_logits, dim=1)  #硬阈值法

            event = self.event_class_values_1d[event]
        R = self.Rnet(R_0,event)

        out = L * R
        #out = self.body(x)


        return out,L,R,L_1,R_1,event_logits



# class EventFormer_Bilevel(nn.Module):
#     def __init__(self, in_channels=3, voxel_channel=16,out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1],mode='train',path=None,level=2):
#         super(EventFormer_Bilevel, self).__init__()
#         self.stage = stage
#         self.mode = mode
#
#         self.Lnet = Denoiser_nolu(in_dim=1, out_dim=1, dim=n_feat, level=level,
#                                       num_blocks=num_blocks)
#
#         self.Rnet = Denoiser_fusion(in_dim=in_channels, out_dim=out_channels,voxel_dim=voxel_channel, dim=n_feat, level=level,
#                                       num_blocks=num_blocks)
#         self.Enet = Denoiser_nolu_R
#         self.Decom_low = Decom()
#         self.Decom_high = Decom()
#         self.model_Decom_low = load_initialize(self.Decom_low, path)
#         high_path  = path.replace("init_low", "init_high")
#         self.model_Decom_high = load_initialize(self.Decom_high, high_path)
#
#     def forward(self, x,event,x_high=None):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         B, Bins, H, W = event.shape
#         input = x
#         if self.mode == 'train':
#             R_0,L_0 = self.model_Decom_low(input)
#             R_1,L_1 = self.model_Decom_high(x_high)
#         else:
#             R_0,L_0 = self.model_Decom_low(input)
#             # R_1,L_1 = torch.ones_like(L_0), torch.ones_like(R_0)
#             R_1, L_1 = self.model_Decom_high(x_high)
#
#         L = self.Lnet(L_0)
#
#         R,event = self.REnet(R_0,event)
#         out = L * R
#         #out = self.body(x)
#
#         event = event.view(B, 3, Bins, H, W)
#         return out,L,R,L_1,R_1,event

def measure_inference_speed(model, dummy_input,event_input, device='cuda', iterations=100, warm_up=20):
    """
    通用测速函数
    :param model: 模型
    :param dummy_input: 虚拟输入数据 (Tensor 或 Dict)
    :param device: 设备
    :param iterations: 正式测速循环次数
    :param warm_up: 预热次数
    """
    model.eval()

    # 1. 预热 (Warm-up)
    # 让 GPU 完成初始化、缓存分配和 kernel 优化
    print(f"正在预热 GPU ({warm_up} 次)...")
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(dummy_input,event_input,dummy_input)

    # 确保预热结束
    if device == 'cuda':
        torch.cuda.synchronize()

    # 2. 正式测速
    print(f"开始测速 (循环 {iterations} 次)...")

    # 使用 torch.cuda.Event 进行更精确的 GPU 计时
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []  # 记录每次的时间

    with torch.no_grad():
        for _ in range(iterations):
            starter.record()

            # === 模型推理 ===
            _ = model(dummy_input,event_input,dummy_input)
            # =============

            ender.record()

            # 等待当前 batch 结束
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 单位是 毫秒(ms)
            timings.append(curr_time)

    # 3. 计算统计数据
    timings = np.array(timings)
    avg_latency = np.mean(timings)  # 平均延迟 (ms)
    std_latency = np.std(timings)  # 标准差

    # 计算 FPS
    # 注意：如果 dummy_input 的 batch_size > 1，需要乘以 batch_size
    # 这里我们假设 dummy_input 是字典，取其中一个 tensor 来判断 batch size
    if isinstance(dummy_input, dict):
        batch_size = list(dummy_input.values())[0].shape[0] if isinstance(list(dummy_input.values())[0],
                                                                          torch.Tensor) else 1
        # 如果 list(dummy_input.values())[0] 是 list (比如 ill_list), 再取一层
        if isinstance(list(dummy_input.values())[0], list):
            batch_size = list(dummy_input.values())[0][0].shape[0]
    else:
        batch_size = dummy_input.shape[0]

    fps = (1000.0 / avg_latency) * batch_size

    print("=" * 40)
    print(f"Batch Size: {batch_size}")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"FPS: {fps:.2f}")
    print("=" * 40)

    return fps, avg_latency

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import torch

    model = EventFormer_Bilevel_event_fusion(path='G:\work\code\low_level\LLIE\Retinexformer_yuanban\ckpt\init_low.pth').cuda().eval()
    print(model)
    inputs = torch.randn((1, 3, 256, 256), device="cuda")
    event = torch.randn((1, 16, 256, 256), device="cuda")
    measure_inference_speed(model,inputs,event)
    flops = FlopCountAnalysis(model, (inputs,event,inputs))

    n_param = sum(p.numel() for p in model.parameters())

    # fvcore 的 total() 是 MACs；常见口径 1 MAC ≈ 2 FLOPs
    gflops = 2.0 * flops.total() / 1e9
    params_m = n_param / 1e6

    print(f"GFLOPs (forward): {gflops:.3f} G")
    print(f"Params: {params_m:.3f} M")
    #out ,_= model(inputs,event,inputs)
    #print(out.shape)