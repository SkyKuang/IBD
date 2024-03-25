from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from losses import trades_loss

class TRADES_Module():
    def __init__(self, basic_net, args,  aux_net=None):
        super(TRADES_Module, self).__init__()
        self.basic_net = basic_net
        self.aux_net = aux_net
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.num_steps = 10
        self.step_size = 2.0/255
        self.epsilon = 8.0/255
        self.restarts = 1
        self.beta = 6.0
        self.norm = 'l_inf'
        self.num_classes = 10

    def train(self, epoch, inputs, targets, index, optimizer):
        #### generating adversarial examples stage
        self.basic_net.eval()
        batch_size = len(inputs)
        logits_nat, logits_adv, loss_trades, loss_nat, loss_adv = trades_loss(model= self.basic_net,
                           x_natural=inputs,
                           y=targets,
                           optimizer=optimizer,
                           step_size=self.step_size,
                           epsilon=self.epsilon,
                           perturb_steps=self.num_steps,
                           beta=self.beta,
                           distance='l_inf')

        #### adversarial tarining stage
        self.basic_net.train()
        self.basic_net.zero_grad()
        optimizer.zero_grad()

        loss = loss_trades
        loss.backward()
        optimizer.step()

        return logits_nat.detach(), logits_adv.detach(), loss.item(), loss_nat.item(), loss_adv.item()

    def test(self, inputs, targets, adversary=None):
        if adversary is not None:
            inputs = adversary.attack(inputs, targets).detach()

        self.basic_net.eval()
        logits = self.basic_net(inputs)
        loss = self.criterion(logits, targets)

        return logits.detach(), loss.item()
