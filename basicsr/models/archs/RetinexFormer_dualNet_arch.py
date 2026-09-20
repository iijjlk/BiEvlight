import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

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

class Denoiser_nolu_R(nn.Module):  #去除了光照分支
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser_nolu_R, self).__init__()
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
        out = self.mapping(fea) + x
        out = torch.sigmoid(out)
        return out
class RetinexFormer_Single_Stage_L(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage_L, self).__init__()
        #self.estimator = Illumination_Estimator(n_feat)  # 修改为分离net
        self.denoiser = Denoiser_nolu(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                      num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, illu_map):

        output_img = self.denoiser(illu_map)

        return output_img


class RetinexFormer_Single_Stage_R(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage_R, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)  # 修改为分离net
        self.denoiser = Denoiser_nolu_R(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                      num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, img):

        output_img = self.denoiser(img)

        return output_img


class RetinexFormer_Retinex(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1],mode='train',path=None):
        super(RetinexFormer_Retinex, self).__init__()
        self.stage = stage
        self.mode = mode
        Rnet_body = [RetinexFormer_Single_Stage_R(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]

        Lnet_body = [
            RetinexFormer_Single_Stage_L(in_channels=1, out_channels=1, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]

        self.Decom_low = Decom()
        self.Decom_high = Decom()
        self.model_Decom_low = load_initialize(self.Decom_low, path)
        high_path  = path.replace("init_low", "init_high")
        self.model_Decom_high = load_initialize(self.Decom_high, high_path)

        self.Rnet = nn.Sequential(*Rnet_body)
        self.Lnet = nn.Sequential(*Lnet_body)
    
    def forward(self, x,x_high=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        input = x
        if self.mode == 'train':
            R_0,L_0 = self.model_Decom_low(input)
            R_1,L_1 = self.model_Decom_high(x_high)
        else:
            R_0,L_0 = self.model_Decom_low(input)
            # R_1,L_1 = torch.ones_like(L_0), torch.ones_like(R_0)
            R_1, L_1 = self.model_Decom_high(x_high)
        L = self.Lnet(L_0)
        R = self.Rnet(R_0)
        out = L * R
        #out = self.body(x)

        return out,L,R,L_1,R_1


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import torch

    model = RetinexFormer_Retinex(stage=1, n_feat=40, num_blocks=[1, 2, 2]).cuda().eval()
    print(model)
    inputs = torch.randn((1, 3, 256, 256), device="cuda")
    flops = FlopCountAnalysis(model, inputs)

    n_param = sum(p.numel() for p in model.parameters())

    # fvcore 的 total() 是 MACs；常见口径 1 MAC ≈ 2 FLOPs
    gflops = 2.0 * flops.total() / 1e9
    params_m = n_param / 1e6

    print(f"GFLOPs (forward): {gflops:.3f} G")
    print(f"Params: {params_m:.3f} M")
