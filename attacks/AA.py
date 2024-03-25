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
from .autoAttack import *

class Attack_Auto():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Auto, self).__init__()
        self.basic_net = basic_net
        self.norm = config['norm']
        self.epsilon = config['epsilon']
        self.verbose = config['verbose']
        self.log_path = config['log_path']
        self.version = config['version']
        self.individual = config['individual']
        self.is_tf_model = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.basic_net.eval()
        self.adversary = AutoAttack(self.basic_net, norm=self.norm, eps=self.epsilon, log_path=self.log_path,
            version=self.version, device=device, verbose=self.verbose)

    def attack(self,
                inputs,
                targets,):

        x = inputs.detach()
        if not self.individual:
            x_adv = self.adversary.run_standard_evaluation(x, targets, bs=100)
        else:
            x_adv = self.adversary.run_standard_evaluation_individual(x, targets, bs=100)
        return x_adv