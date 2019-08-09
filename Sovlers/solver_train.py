import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Data import get_loader
from Models import get_model

class TrainSolver(object):

    def __init__(self, config):

        self.config = config
        self.cfg_solver = config['solver']
        self.cfg_dataset = config['data']
        self.cfg_model = config['model']
        self.cfg_solver = config['solver']
        self.cuda = config['cuda']

    def run(self):
        train_loader, val_loader = get_loader(self.config)
        model = get_model(self.config)

        if self.cuda:
            model = nn.DataParallel(model)
            model.cuda()
        
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        
        if self.cfg_solver['optimizer_type'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg_solver['lr_init'])
        
        else:
            raise NotImplementedError('Optimizer type [{:s}] is not supported'.format(self.cfg_solver['optimizer_type']))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg_solver['milestones'], gamma=0.5)
        counter = 1
        for epoch in range(1, self.cfg_solver['max_steps'] + 1):
            for batch_idx,  data_batch in enumerate(train_loader):
                start_time = time.time()
                (imgL, imgR, disp_L, scale) = data_batch

                if self.cuda:
                    imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()
                
                mask = disp_L < self.cfg_model['max_disp']
                mask.detach_()
                
                optimizer.zero_grad()

                disp_pred = model(imgL, imgR)
                loss = F.smooth_l1_loss(disp_pred[mask], disp_L[mask], size_average=True)
                
                loss.backward()
                optimizer.step()

                print('Iter %d training loss = %.3f , time = %.2f' %(counter, loss.data[0], time.time() - start_time))
                counter += 1
                scheduler.step()
            
        print('1')

        