import torch
import torch.nn as nn
import torch.nn.functional as F


def _gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2).to(input_tensor.device)
    smooth_kernel_y = smooth_kernel_x.transpose(2, 3)
    kernel = smooth_kernel_x if direction == "x" else smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    return grad_out


def _ave_gradient(input_tensor, direction):
    return F.avg_pool2d(_gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)


def smooth_loss(input1, input2):
    # 如果是灰度事件栈可直接使用单通道
    if input2.shape[1] == 3:  # RGB to gray
        input2 = 0.299 * input2[:,0,:,:] + 0.587 * input2[:,1,:,:] + 0.114 * input2[:,2,:,:]
        input2 = input2.unsqueeze(1)
    return torch.mean(
        _gradient(input1, "x") * torch.exp(-10 * _ave_gradient(input2, "x")) +
        _gradient(input1, "y") * torch.exp(-10 * _ave_gradient(input2, "y"))
    )

