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

class Attack_JSMA():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_JSMA, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.num_classes = config['num_classes']
        self.basic_net.eval()

        from advertorch.attacks import JSMA
        self.adversary = JSMA(self.basic_net, num_classes=self.num_classes,
                 clip_min=self.clip_min, clip_max=self.clip_max, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                 theta=1.0, gamma=0.1, comply_cleverhans=False)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv