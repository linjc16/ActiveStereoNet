import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Data import get_loader
from Models import get_model
from Losses import get_losses
from Metrics.metrics import epe_metric
from Metrics.metrics import tripe_metric
import pdb


class TestSolver(object):
    def __init__(self, config):
        self.config = config
        self.cfg_solver = config['solver']
        self.cfg_dataset = config['data']
        self.cfg_model = config['model']
        
        self.max_disp = self.cfg_model['max_disp']
        self.model = get_model(self.config)
        self.test_loader = get_loader(self.config)
        
    
    def load_checkpoint(self):
        ckpt_root = os.path.join(self.cfg_solver['exp_prefix'], self.cfg_solver['model_name'], 'models')
        ckpt_name = 'iter_{:d}.pth'.format(self.cfg_solver['resume_iter'])
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(states['model_state'])


    def run(self):
        self.model.cuda()

        if self.cfg_solver['resume_iter'] > 0:
            self.load_checkpoint()
            print('Model loaded.')
        
        self.model.eval()

        start_time = time.time()
        with torch.no_grad():
            EPE_metric = 0.0
            TriPE_metric = 0.0
            N_total = 0.0
            pdb.set_trace()
            for test_batch in self.test_loader:
                imgL, imgR, disp_L, _ = test_batch
                imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

                N_curr = imgL.shape[0]

                disp_pred = self.model(imgL, imgR)

                EPE_metric = (epe_metric(disp_L, disp_pred, self.max_disp) * N_curr + EPE_metric * N_total) / (N_curr + N_total)
                TriPE_metric = (tripe_metric(disp_L, disp_pred, self.max_disp) + TriPE_metric * N_total) / (N_curr + N_total)

                N_total += N_curr
        

        elapsed = time.time() - start_time
        print(
            'Test: EPE = {:.6f} px, 3PE = {:.3f} %, time = {:.3f} s.'.format(
                EPE_metric, TriPE_metric * 100, elapsed / N_total
            )
        )




