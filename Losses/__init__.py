from .supervise import *

def get_losses(name, **kwargs):

    name = name.lower()
    if name == 'rhloss':
        loss = RHLoss(**kwargs)
    elif name == 'xtloss':
        loss = XTLoss(**kwargs)
    else:
        raise NotImplementedError('Loss [{:s}] is not supported.'.format(name))
    
    return loss
