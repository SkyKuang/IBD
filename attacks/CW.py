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

class Attack_CW():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_CW, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.max_iters = config['num_steps']
        self.num_classes = config['num_classes']
        self.basic_net.eval()

        from advertorch.attacks import CarliniWagnerL2Attack
        self.adversary = CarliniWagnerL2Attack(self.basic_net, num_classes=self.num_classes, confidence=0,
                targeted=False, learning_rate=0.01,
                binary_search_steps=9, max_iterations=self.max_iters,
                abort_early=True, initial_const=1e-3,
                clip_min=self.clip_min, clip_max=self.clip_max, loss_fn=None)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv