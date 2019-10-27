import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def epe_metric(target, output, maxdisp):
    mask = (target < maxdisp).float()
    
    target *= mask
    output *= mask
    
    return torch.abs(target - output).mean()

#def epe_metric(target, output, maxdisp):
#    mask = target < maxdisp
#    return torch.abs(output[mask] - target[mask]).mean()

    
#def tripe_metric(target, output, maxdisp):
    #mask = (target < maxdisp).float()
    
    #target *= mask
    #output *= mask

    #delta = torch.abs(target - output)
    #gt3 = (delta > 3.0).float()
    #eps = 1e-7
    #return gt3.sum() / (delta.numel() + eps)

def tripe_metric(target, output, maxdisp):
    #pdb.set_trace()
    delta = torch.abs(target - output)
    correct = ((delta < 3) | torch.lt(delta, target * 0.05))
    eps = 1e-7
    return 1 - (float(torch.sum(correct))/(delta.numel() + eps))
