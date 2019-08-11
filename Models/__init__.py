from .ActiveStereoNet import ActiveStereoNet

def get_model(config):

    cfg_net = config['model']
    net_name = cfg_net['which_model'].lower()
    if net_name == 'activestereonet':
        max_disp = cfg_net['max_disp']
        scale_factor = cfg_net['scale_factor']
        img_shape = config['data']['crop_size']
        model = ActiveStereoNet(max_disp, scale_factor, img_shape)
    else:
        raise NotImplementedError('Model [{:s}] is not supported.'.format(net_name))

    return model