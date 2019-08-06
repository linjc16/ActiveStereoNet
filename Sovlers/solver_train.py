import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Data import get_loader
from Models import get_model

class TrainSolver(object):

    def __init__(self, config):

        self.config = config
        self.cfg_solver = config['solver']
    