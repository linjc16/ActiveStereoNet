import torch
import torch.nn as nn
import torch.nn.functional as F

class RHLoss(nn.Module):

    def __init__(self, max_disp):

        super(RHLoss, self).__init__()
        self.max_disp = max_disp
        self.crit = nn.SmoothL1Loss()
    
    def forward(self, output, target):

        mask = (target < self.max_disp).float()
        output *= mask
        target *= mask
        loss = self.crit(output, target)
        
        return loss