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

class Attack_BPDA():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_BPDA, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        self.basic_net.eval()

        from advertorch.defenses import MedianSmoothing2D
        from advertorch.defenses import BitSqueezing
        from advertorch.defenses import JPEGFilter
        from advertorch.attacks import LinfPGDAttack
        from advertorch.bpda import BPDAWrapper

        bits_squeezing = BitSqueezing(bit_depth=5)
        median_filter = MedianSmoothing2D(kernel_size=3)
        jpeg_filter = JPEGFilter(10)

        defense = nn.Sequential(
            jpeg_filter,
            bits_squeezing,
            median_filter,
            )
        defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
        defended_model = nn.Sequential(defense_withbpda, self.basic_net)
        self.adversary = LinfPGDAttack(
            defended_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.epsilon,
            nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
            targeted=self.targeted)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv

class Attack_BPDA_AEP():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_BPDA_AEP, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        self.basic_net.eval()

        from advertorch.attacks import LinfPGDAttack
        from advertorch.bpda import BPDAWrapper
        from defense import AEP

        AEP_onfig = {
            'epsilon': 8/255,
            'num_steps': 5,
            'step_size': 2/255,
            }
        aep = AEP(basic_net, AEP_onfig)

        defense = nn.Sequential(
            aep,
            )

        defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
        defended_model = nn.Sequential(defense_withbpda, self.basic_net)
        self.adversary = LinfPGDAttack(
            defended_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=self.epsilon,
            nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
            targeted=self.targeted)

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv