'''Some utility functions
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch
import math
import torch.nn.functional as F

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

def get_acc(outputs, targets):
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    acc = 1.0 * correct / total
    return acc

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class logger(object):
    def __init__(self, path, file_name):
        self.path = path
        self.file_name = file_name

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, f"{self.file_name}.log"), 'a') as f:
            f.write(msg + "\n")

def load_checkpoint(basic_net, f_path):
    if os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        if checkpoint['net'] is not None:
            basic_net.load_state_dict(checkpoint['net'])
            epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            optimizer = checkpoint['optimizer']
            print(f'==> load epoch in {epoch}')
        else:
            basic_net.load_state_dict(checkpoint)
    else:
        raise NotImplementedError("no checkpoint directory or file found!")
    return True

def save_image(data, path,row=5):
    # data : list of image
    # path : save path
    torchvision.utils.save_image(data, path, nrow=row, normalize=True,padding=0)

def get_schedule(args):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.max_epoch * 2 // 5, args.max_epoch], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'mylr':
        def lr_schedule(t):
            if t < args.decay_epoch1:
                return args.lr_max
            elif t < args.decay_epoch2:
                if (t+1) ==args.decay_epoch1:
                    return 1
                else:
                    v = (2 / (1 + math.exp(-4 * (t - args.decay_epoch1) / (args.max_epoch - args.decay_epoch1))) - 1)
                    return args.lr_max * (1-v)
            else:
                return 0.0001
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < args.decay_epoch1:
                return args.lr_max
            elif t < args.decay_epoch2:
                return args.lr_max / 10.
            elif t < args.decay_epoch3:
                return args.lr_max / 100.
            else:
                return 0.0001
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.max_epoch // 3, args.max_epoch * 2 // 3, args.max_epoch], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.decay_epoch1:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.max_epoch//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.max_epoch * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.max_epoch, args.max_epoch], [0, args.lr_max, 0])[0]
        
    return lr_schedule

def gen_rand_labels(y, num_classes):
    targets = torch.randint_like(y, low=0, high=num_classes)
    for i in range(len(targets)):
        while targets[i]==y[i]:
            targets[i] = torch.randint(low=0, high=10, size=(1,))
    return targets

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def one_hot_tensor(y_batch_tensor, num_classes):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def label_smoothing(y_batch_tensor, num_classes, delta):
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * y_batch_tensor + delta / (num_classes - 1)
    return y_batch_smooth

class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        # return F.mse_loss(p, z.detach(), dim=-1).mean()
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception