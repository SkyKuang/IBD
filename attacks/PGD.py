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
from attacks import CWLoss

class Attack_PGD():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter = config['num_steps']
        self.targeted = config['targeted']
        if config['loss_fn'] != None:
            self.loss_fn = config['loss_fn']
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.num_classes = config['num_classes']
        self.norm = config['norm']
        self.basic_net.eval()

        if self.norm == 'Linf':
            from advertorch.attacks import LinfPGDAttack
            self.adversary = LinfPGDAttack(
                self.basic_net, loss_fn=self.loss_fn, eps=self.epsilon,
                nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
                targeted=self.targeted)
        elif self.norm == 'L2':
            from advertorch.attacks import L2PGDAttack
            self.adversary = L2PGDAttack(
                self.basic_net, loss_fn=self.loss_fn, eps=self.epsilon,
                nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
                targeted=self.targeted)
        else:
            pass

    def attack(self, inputs, targets):
        if self.targeted:
            targets = gen_rand_labels(targets, self.num_classes)
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv

def gen_rand_labels(y, num_classes):
    targets = torch.randint_like(y, low=0, high=num_classes)
    for i in range(len(targets)):
        while targets[i]==y[i]:
            targets[i] = torch.randint(low=0, high=10, size=(1,))
    return targets

class PGD_CW():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(PGD_CW, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon  = config['epsilon']
        self.eps_iter = config['step_size']
        self.nb_iter  = config['num_steps']
        self.targeted = config['targeted']
        self.num_classes = config['num_classes']
        self.loss_fn = CWLoss(self.num_classes)
        self.norm = config['norm']
        self.basic_net.eval()

        if self.norm == 'Linf':
            from advertorch.attacks import LinfPGDAttack
            self.adversary = LinfPGDAttack(
                self.basic_net, loss_fn=self.loss_fn, eps=self.epsilon,
                nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
                targeted=self.targeted)
        elif self.norm == 'L2':
            from advertorch.attacks import L2PGDAttack
            self.adversary = L2PGDAttack(
                self.basic_net, loss_fn=self.loss_fn, eps=self.epsilon,
                nb_iter=self.nb_iter, eps_iter=self.eps_iter, rand_init=True, clip_min=self.clip_min, clip_max=self.clip_max,
                targeted=self.targeted)
        else:
            pass

    def attack(self, inputs, targets):
        x_adv = self.adversary.perturb(inputs, targets)
        return x_adv

import foolbox
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attack_PGD_FB():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.epsilon = config['epsilon']
        self.step_size = config['step_size']
        self.num_steps = config['num_steps']
        self.targeted = config['targeted']
        self.loss_fn = config['loss_fn']
        self.norm = config['norm']
        self.num_classes = config['num_classes']
        self.basic_net.eval()

        self.fmodel = PyTorchModel(self.basic_net, bounds=(self.clip_min, self.clip_max))
        self.adversary = fa.PGD(steps=self.num_steps, abs_stepsize=self.step_size)

    def attack(self, inputs, targets):
        raw_advs, clipped_advs, success = self.adversary(self.fmodel, inputs, targets, epsilons=self.epsilon)
        return clipped_advs