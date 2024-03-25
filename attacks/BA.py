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
import torchvision.models as models
import eagerpy as ep
import foolbox.attacks as fa
import foolbox
from foolbox import PyTorchModel, accuracy, samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attack_BA():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_BA, self).__init__()
        self.basic_net = basic_net
        self.epsilon = config['epsilon']
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.num_steps = config['num_steps']
        self.basic_net.eval()

        self.fmodel = PyTorchModel(self.basic_net, bounds=(self.clip_min, self.clip_max))
        # self.adversary = fa.BoundaryAttack(steps=1000, spherical_step=0.1, source_step=0.1, source_step_convergance=1e-3)
        self.adversary = fa.BoundaryAttack(steps=5000,
                                        init_attack=fa.LinearSearchBlendedUniformNoiseAttack(steps=50),
                                        update_stats_every_k=1)

    def attack(self, inputs, targets):
        raw_advs, clipped_advs, success = self.adversary(self.fmodel, inputs, targets, epsilons=0.1)
        return clipped_advs


