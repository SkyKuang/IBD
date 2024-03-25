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

class Attack_SPA():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_SPA, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        self.num_classes = config['num_classes']
        self.basic_net.eval()

        from advertorch.attacks import SpatialTransformAttack
        self.adversary = SpatialTransformAttack(self.basic_net, num_classes=self.num_classes, confidence=0,
                initial_const=1, max_iterations=self.nb_iter, search_steps=1,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                clip_min=self.clip_min, clip_max=self.clip_max, abort_early=True, targeted=self.targeted)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv