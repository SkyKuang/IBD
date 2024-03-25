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

class Attack_MIM():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_MIM, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        self.basic_net.eval()

        from advertorch.attacks import MomentumIterativeAttack
        self.adversary = MomentumIterativeAttack(
            self.basic_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.epsilon,
            nb_iter=self.nb_iter, eps_iter=self.eps_iter, clip_min=self.clip_min, clip_max=self.clip_max,
            targeted=self.targeted, decay_factor=1.)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv