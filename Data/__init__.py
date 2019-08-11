import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from .SceneFlow import SceneFlowDataset

def get_loader(config):

    dset = config['dataset_name'].lower()
    if dset == 'sceneflow':
        return get_scene_flow_loader(config)
    else:
        raise NotImplementedError('Dataset [{:s}] is not supported.'.format(dset))

def get_scene_flow_loader(config):

    cfg_mode = config['mode'].lower()
    
    if cfg_mode == 'train':
        train_loader = DataLoader(
            create_scene_flow_dataset(config['data'], 'train'),
            batch_size=config['solver']['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            create_scene_flow_dataset(config['data'], 'val'),
            batch_size=config['solver']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        return train_loader, val_loader
    elif cfg_mode == 'test':
        test_loader = DataLoader(
            create_scene_flow_dataset(config['data'], 'test'),
            batch_size=config['solver']['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        return test_loader
    else:
        raise NotImplementedError('Mode [{:s}] is not supported.'.format(cfg_mode))

def create_scene_flow_dataset(cfg_data, mode):
    
    data_root = cfg_data['data_root']
    npy_root = cfg_data['npy_root']
    test_split = cfg_data['test_split']
    val_split = cfg_data['val_split']
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    return SceneFlowDataset(data_root, npy_root, val_split, test_split, transform, mode)