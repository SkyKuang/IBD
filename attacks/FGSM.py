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

class Attack_FGSM():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FGSM, self).__init__()
        self.basic_net = basic_net
        self.epsilon = config['epsilon']
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.targeted = config['targeted']
        self.targeted = config['targeted']
        self.basic_net.eval()
        from advertorch.attacks import FGSM
        self.adversary = FGSM(
            self.basic_net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps= self.epsilon,
            clip_min=0.0, clip_max=1.0, targeted=False)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv