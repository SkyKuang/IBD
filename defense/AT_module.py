from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import uniform

from torch import optim


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from utils import clamp
from utils import label_smoothing, one_hot_tensor, softCrossEntropy

class AT_Module():
    def __init__(self, basic_net, args, aux_net=None):
        super(AT_Module, self).__init__()
        self.basic_net = basic_net
        self.aux_net = aux_net
        self.criterion = nn.CrossEntropyLoss()
        self.num_steps = 10
        self.step_size = 2.0/255
        self.epsilon = 8.0/255
        self.restarts = 1
        self.norm = "l_inf"
        self.early_stop = False
        self.alp = False
        if self.alp:
            print('adversarial logits pairs!!!!!!!!')

    def train(self, epoch, inputs, targets, index, optimizer):
        #### generating adversarial examples stage
        self.basic_net.eval()
        adv_inputs = attack_pgd(self.basic_net, inputs, targets,attack_iters=self.num_steps, epsilon=self.epsilon,
                                alpha=self.step_size,restarts=self.restarts,norm=self.norm)

        #### adversarial tarining stage
        self.basic_net.train()
        self.basic_net.zero_grad()
        optimizer.zero_grad()

        logits_nat = self.basic_net(inputs)
        loss_nat = self.criterion(logits_nat, targets)

        logits_adv = self.basic_net(adv_inputs)
        loss_adv = self.criterion(logits_adv, targets)

        # loss_sf_ce = softCrossEntropy()
        # y_gt = one_hot_tensor(targets, 10)
        # targets_ls = label_smoothing(y_gt, y_gt.size(1), 0.3)
        # loss_adv = loss_sf_ce(logits_adv, targets_ls)

        ####  alp tarin model 
        if self.alp:
            loss_alp = F.mse_loss(logits_nat, logits_adv)
            loss = (loss_adv + loss_nat) * 0.5 + loss_alp
        else:
            loss = loss_adv

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

upper_limit, lower_limit = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.uniform_(-0.5,0.5).renorm(p=2, dim=1, maxnorm=epsilon)
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    x_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
    return x_adv