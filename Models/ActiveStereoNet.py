import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .blocks import *



class SiameseTower(nn.Module):
    def __init__(self, scale_factor):
        super(SiameseTower, self).__init__()

        self.conv1 = conv_block(nc_in=3, nc_out=32, k=3, s=1, norm=None, act=None)
        resblocks = [ResBlock(32, 32, 3, 1, 1)] * 3
        self.res_blocks = nn.Sequential(*res_blocks)    
        convblocks = [conv_block(32, 32, k=2, s=1, norm='bn', act='lrelu')] * scale_factor
        self.conv_blocks = nn.Sequential(*convblocks)
        self.conv2 = conv_block(nc_in=32, nc_out=32, k=3, s=1, norm=None, act=None)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv_blocks(out)
        out = self.conv2(out)

        return out

class CoarseNet(nn.Module):
    def __init__(self, maxdisp, scale_factor):
        super(CoarseNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.conv3d_1 = conv3d_block(64, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_2 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_3 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')

        self.conv3d_4 = conv3d_block(32, 1, 3, 1, norm=None, act=None)

        

    def forward(self, refimg_fea, targetimg_fea):
        '''
        Args:
            refimg_fea: output of SiameseTower for a left image
            targetimg_fea: output of SiameseTower for the right image

        '''
        #Cost Volume
        cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//self.scale_factor, refimg_fea.size()[2], refimg_fea.size()[3]).to(self.device)
        
        for i in range(self.maxdisp//self.scale_factor):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
            
        cost = cost.contiguous()
        
        cost0 = self.conv3d_1(cost0)
        cost0 = self.conv3d_2(cost0) + cost0
        cost0 = self.conv3d_3(cost0) + cost0
        
        cost = self.conv3d_4(cost0)
        cost = F.upsample(
            cost, 
            [self.maxdisp, refimg_fea.size()[2] * self.scale_factor, refimg_fea.size()[3] * self.scale_factor],
            mode='bilinear'
        )
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)

        return pred

        
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # stream_1, left_img
        self.conv1_s1 = conv_block(3, 16, 3, 1)
        self.resblock1_s1 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s1 = ResBlock(16, 16, 3, 1, 2)

        # stream_2, upsampled low_resolution disp
        self.conv1_s2 = conv_block(1, 16, 1, 1)
        self.resblock1_s2 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s2 = ResBlock(16, 16, 3, 1, 2)

        # cat
        self.resblock3 = ResBlock(32, 32, 3, 1, 4)
        self.resblock4 = ResBlock(32, 32, 3, 1, 8)
        self.resblock5 = ResBlock(32, 32, 3, 1, 1)
        self.resblock6 = ResBlock(32, 32, 3, 1, 1)
        self.conv2 = conv_block(32, 1, 3, 1)

    def forward(self, left_img, up_disp):
        stream1 = self.conv1_s1(left_img)
        stream1 = self.resblock1_s1(stream1)
        stream1 = self.resblock2_s1(stream1)

        stream2 = self.conv1_s2(up_disp)
        stream2 = self.resblock1_s2(stream2)
        stream2 = self.resblock2_s2(stream2)

        out = torch.cat((stream1, stream2), 1)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)
        out = self.conv2(out)

        return out

        
class InvalidationNet(nn.Module):
    def __init__(self):
        super(InvalidationNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        resblocks1 = [ResBlock(64, 64, 3, 1, 1)] * 5
        self.resblocks1 = nn.Sequential(*resblocks1)
        self.conv1 = conv_block(64, 1, 3, 1, norm=None, act=None)

        self.conv2 = conv_block(5, 32, 3, 1)
        resblocks2 = [ResBlock(32, 32, 3, 1, 1)] * 4
        self.resblocks2 = nn.Sequential(*resblocks2)
        self.conv3 = conv_block(32, 1, 3, 1, norm=None, act=None)

    def forward(self, left_tower, right_tower, left_img, freso_disp):

        features = torch.cat((left_tower, right_tower), 1)
        out1 = self.resblocks1(features)
        out1 = self.conv1(out1)

        input = torch.cat((left_img, out1, freso_disp), 1)
        
        out2 = self.conv2(input)
        out2 = self.resblocks2(out2)
        out2 = self.conv3(out2)

        return out2
        


class ActiveStereoNet(nn.Module):
    def __init__(self, maxdisp, scale_factor):
        super(ActiveStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.SiameseTower = SiameseTower(scale_factor)
        self.CoarseNet = CoarseNet(maxdisp, scale_factor)
        self.RefineNet = RefineNet()
        #self.InvalidationNet = InvalidationNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, left, right):
        left_tower = self.SiameseTower(left).cuda()
        right_tower = self.SiameseTower(right).cuda()

        coarseup_pred = self.CoarseNet(left_tower, right_tower).cuda()
        disp = self.RefineNet(left, coarseup_pred).cuda()

        return disp + coarseup_pred




class XTLoss(nn.Module):
    '''
    Args:
        left_img right_img: N * C * H * W,
        dispmap : N * H * W
    '''
    def __init__(self):
        super(XTLoss, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.theta = torch.Tensor(
            [[1, 0, 0],  # 控制左右，-右，+左
            [0, 1, 0]]    # 控制上下，-下，+上
        )
        self.inplanes = 3
        self.outplanes = 3
        

    def forward(self, left_img, right_img, dispmap):

        n, c, h, w = left_img.shape

        
        self.theta = self.theta.repeat(left_img.size()[0], 1, 1)

        grid = F.affine_grid(self.theta, left_img.size())
        
        dispmap_norm = dispmap * 2 / w
        dispmap_norm = torch.from_numpy(dispmap_norm).unsqueeze(3).to(self.device)
        dispmap_norm = torch.cat((dispmap_norm, torch.zeros(dispmap_norm.size()).to(self.device)), dim=3).to(self.device)

        grid -= dispmap_norm
        
        recon_img = F.grid_sample(right_img, grid)

        recon_img_LCN, _, _ = self.LCN(recon_img, 9)

        left_img_LCN, _, left_std_local = self.LCN(left_img, 9)
        
        losses = torch.abs(((left_img_LCN - recon_img_LCN) * left_std_local)).mean().to(self.device)
        
        return losses


    def LCN(self, img, kSize):
        '''
            Args: 
                img : N * C * H * W
                kSize : 9 * 9
        '''

        w = torch.ones((self.outplanes, self.inplanes, kSize, kSize)).to(self.device) / (kSize * kSize)
        mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2)

        mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2)
        std_local = (mean_square_local - mean_local ** 2) * (kSize ** 2) / (kSize ** 2 - 1)
        
        epsilon = 1e-6
        
        return (img - mean_local) / (std_local + epsilon), mean_local, std_local
