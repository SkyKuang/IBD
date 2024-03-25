import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

train_attack_config = {
    'clip_min':0,
    'clip_max':1.0,
    'targeted': False,
    'random_start': True,
    'epsilon': 8.0/255,
    'num_steps': 10,
    'step_size': 2.0/255,
    'num_classes': 10,
    'norm': 'Linf',
    'loss_fn':None,
}

eval_attack_config = {
    'clip_min':0,
    'clip_max':1.0,
    'targeted': False,
    'random_start': True,
    'epsilon': 8.0/255,
    'num_steps': 10,
    'step_size': 2.0/255,
    'num_classes': 10,
    'norm': 'Linf',
    'loss_fn':None,
}

attack_config = {
    'clip_min':0,
    'clip_max':1.0,
    'targeted': False,
    'random_start': True,
    'epsilon': 8.0/255,
    'num_steps': 100,
    'step_size': 2.0/255,
    'num_classes': 10,
    'norm': 'Linf',
    'individual': False,
    'log_path': './log/log.txt',
    'version': '3',
    'verbose': False,
    'loss_fn': None,
}
