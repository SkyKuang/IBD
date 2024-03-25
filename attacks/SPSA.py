from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.autograd import Variable
import utils
import math
import pdb

class Attack_SPSA():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_SPSA, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        self.num_classes = config['num_classes']
        self.basic_net.eval()

        from advertorch.attacks import  LinfSPSAAttack
        self.adversary = LinfSPSAAttack(self.basic_net, self.epsilon, delta=0.01, lr=0.01, nb_iter=10,
                 nb_sample=128, max_batch_size=64, targeted=False,
                 loss_fn=None, clip_min=0.0, clip_max=1.0)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv