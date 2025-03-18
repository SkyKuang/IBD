from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numpy.core.numeric import True_
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from utils import label_smoothing, one_hot_tensor,softCrossEntropy
import torchvision
import math
from torch.autograd import Variable
from utils import clamp
from losses import pearson_loss, sinkhorn_loss_joint_IPOT
import pdb
import argparse
import numpy as np
import os

def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes

class Dist_Module(nn.Module):
    def __init__(self, basic_net, args, aux_net=None):
        super(Dist_Module, self).__init__()
        self.basic_net = basic_net
        self.aux_net = aux_net
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.num_steps = 10
        self.step_size = 2.0/255
        self.epsilon = 8.0/255
        self.restarts = 1
        self.norm = "l_inf"
        self.early_stop = False
        self.args = args
        self.num_classes = args.num_classes

        self.beta = 1
        self.qk_dim = 128
        self.guide_layers = np.arange(1, (34 - 4) // 2 + 1)
        # self.guide_layers  = np.arange(1, (18 - 2) // 2 + 1)
        self.hint_layers  = np.arange(1, (18 - 2) // 2 + 1)
        with torch.no_grad():
            data = torch.randn(2, 3, 32, 32).cuda()
            _, feat_t = self.aux_net(data, is_feat=True)
            _, feat_s = self.basic_net(data, is_feat=True)
        self.s_shapes = [feat_s[i].size() for i in self.hint_layers]
        self.t_shapes = [feat_t[i].size() for i in self.guide_layers]
        self.n_t, self.unique_t_shapes = unique_shape(self.t_shapes)
        self.IBD = IBD(self).cuda()
        self.aux_optimizer = optim.SGD(self.IBD.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        from models import get_ResNet_F
        proxy_net = get_ResNet_F(depth=18,num_classes=self.num_classes).cuda()
        proxy_net = proxy_net.cuda()
        save_point = args.model_dir+args.dataset+os.sep
        aux_path = save_point + f'cifar10-natural-f-best.t7'
        # aux_path = save_point + args.aux_name + f'-latest.t7'
        if os.path.isfile(aux_path):
            checkpoint = torch.load(aux_path)
            pretrain_dict = checkpoint['net']
            # pretrain_dict = checkpoint
            model_dict = {}
            state_dict = proxy_net.state_dict()
            for k, v in pretrain_dict.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            proxy_net.load_state_dict(state_dict)
            proxy_net.eval()
        self.proxy_net = proxy_net

    def train(self, epoch, inputs, targets, index, optimizer):
        #### generating adversarial examples stage
        for param_group in self.aux_optimizer.param_groups:
            if epoch < self.args.decay_epoch1:
                param_group['lr'] = 0.1
            elif epoch < self.args.decay_epoch2:
                param_group['lr'] = 0.01
            else:
                param_group['lr'] = 0.001
        self.basic_net.eval()
        self.aux_net.eval()
        batch_size = len(inputs)
        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        logits_tea, feat_list_tea = self.aux_net(x_adv, True)
        feat_list_tea = [f.detach() for f in feat_list_tea]
        logits_tea = logits_tea.detach()
        if self.norm == 'l_inf':
            for _ in range(self.num_steps): 
                x_adv.requires_grad_()
                with torch.enable_grad():
                    logits_adv = self.basic_net(x_adv)
                    #############
                    loss_adv = self.criterion_kl(F.log_softmax(logits_adv, dim=1),
                                        F.softmax(logits_tea, dim=1))
               
                loss_adv = loss_adv 
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon), inputs + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        adv_inputs = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        # logits_tea = self.proxy_net(inputs)

        #### adversarial tarining stage
        self.basic_net.train()
        self.basic_net.zero_grad()
        self.IBD.train()
        self.IBD.zero_grad()
        optimizer.zero_grad()
        self.aux_optimizer.zero_grad()

        logits_nat, feat_list_nat = self.basic_net(inputs, True)
        loss_nat = self.criterion(logits_nat, targets)
        logits_adv, feat_list_adv = self.basic_net(adv_inputs, True)
        loss_adv = self.criterion(logits_adv, targets)
        loss_ibd = self.IBD(feat_list_adv, feat_list_tea)

        loss_nat = (1.0/batch_size) * self.criterion_kl(F.log_softmax(logits_nat, dim=1), F.softmax(logits_tea, dim=1))
        loss_adv = (1.0/batch_size) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_tea, dim=1))
        
        a = b =0.8
        loss = loss_nat*(1-a) + loss_adv*a + loss_ibd*b
        loss.backward()
        optimizer.step()
        self.aux_optimizer.step()

        return logits_nat.detach(), logits_adv.detach(), loss.item(), loss_adv.item(), loss_ibd.item()

    def test(self, inputs, targets, adversary=None):
        if adversary is not None:
            inputs = adversary.attack(inputs, targets).detach()

        self.basic_net.eval()
        logits = self.basic_net(inputs)
        loss = self.criterion(logits, targets)

        return logits.detach(), loss.item()

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))

class IBD(nn.Module):
    def __init__(self, args):
        super(IBD, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.attention = Attention(args).cuda()

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args).cuda()
        self.linear_trans_t = LinearTransformTeacher(args).cuda()

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []
        
        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])
        
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s


