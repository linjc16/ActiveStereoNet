from .ActiveStereoNet import ActiveStereoNet

def get_model(config):

    cfg_net = config['model']
    net_name = cfg_net['which_model'].lower()
    if cfg_net == 'activestereonet':
        max_disp = cfg_net['max_disp']
        scale_factor = cfg_net['scale_factor']
        model = ActiveStereoNet(max_disp, scale_factor)
    else:
        raise NotImplementedError('Model [{:s}] is not supported.'.format(net_name))

    return model