import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_block(nc_in, nc_out, k, s, norm='bn', act='lrelu', dilation=1):

    blocks = [
        nn.Conv2d(nc_in, nc_out, k, s, dilation if dilation > 1 else k // 2, dilation=dilation)
    ]
    if norm is not None:
        norm = norm.lower()
        if norm == 'bn':
            blocks.append(nn.BatchNorm2d(nc_out))
        elif norm == 'in':
            blocks.append(nn.InstanceNorm2d(nc_out))
        else:
            raise RuntimeError

    if act is not None:
        act = act.lower()
        if act == 'relu':
            blocks.append(nn.ReLU(True))
        elif act == 'lrelu':
            blocks.append(nn.LeakyReLU(0.2, True))
        else:
            raise RuntimeError
    
    return nn.Sequential(*blocks)


def conv3d_block(in_planes, out_planes, kernel_size, stride, norm='bn', act='lrelu'):

    blocks = [
        nn.Conv3d(in_planes, out_planes, kernel_size, stride, kernel_size // 2)
    ]
    if norm is not None:
        norm = norm.lower()
        if norm == 'bn':
            blocks.append(nn.BatchNorm3d(out_planes))
        elif norm == 'in':
            blocks.append(nn.InstanceNorm3d(out_planes))
        else:
            raise RuntimeError
    
    if act is not None:
        act = act.lower()
        if act == 'lrelu':
            blocks.append(nn.LeakyReLU(0.2, True))
        elif act == 'relu':
            blocks.append(nn.ReLU(True))
        else:
            raise RuntimeError
    
    return nn.Sequential(*blocks)


class ResBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, dilation=1):
        super(ResBlock, self).__init__()
        self.conv = conv_block(in_planes, out_planes, kernel_size, stride, norm='bn', act='lrelu', dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        out += x
        return out



class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.disp = torch.Tensor(
            np.reshape(np.array(range(maxdisp)), 
            [1, maxdisp, 1, 1]
            ).to(self.device),
            requires_grad=False
        )
    
    def forward(self, x):
        disp = self.disp.repeat(x.size([0], 1, x.size()[2], x.size()[3]))
        out = torch.sum(x * disp, 1)
        return out

