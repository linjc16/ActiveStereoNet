import torch
import torch.nn as nn
import torch.nn.functional as F



def epe_metric(target, output, maxdisp):
    mask = (target < maxdisp).float()
    
    target *= mask
    output *= mask

    return torch.abs(target - output).mean()

    
def tripe_metric(target, output, maxdisp):
    mask = (target < maxdisp).float()
    
    target *= mask
    output *= mask

    delta = torch.abs(target - output)
    gt3 = (delta > 3.0).float()
    eps = 1e-7
    return gt3.sum() / (delta.numel() + eps)