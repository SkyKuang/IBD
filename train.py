'''Train Adversarially Robust Models with Dual-label Geometry Dispersion'''
from __future__ import print_function
import time
import numpy as np
import random
import copy
import os
import pdb
import shutil
import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchcontrib.optim import SWA
from torchvision.datasets import cifar

import utils
from config import args_cf, eval_attack_config
from tqdm import tqdm
from models import get_Network
from defense import get_Module
from attacks import get_adversary
from utils import get_hms, AverageMeter, logger, get_schedule
from dataloader import data_loader

torch.manual_seed(111)
torch.cuda.manual_seed(111)
np.random.seed(111)
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)
args = args_cf.get_train_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
if not os.path.isdir(args.model_dir):
    os.mkdir(args.model_dir)
save_point = args.model_dir+args.dataset+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)

shutil.copyfile(f'./defense/{args.net_module}_module.py', os.path.join('./checkpoint/save_script', f'{args.save_name}_module.py'))

SW_PATH = os.path.join(args.model_dir, 'runs'+f'/{args.save_name}')
writer = SummaryWriter(SW_PATH)
log = logger(path=save_point, file_name=args.save_name)
log.info(str(args))
log.info(str(eval_attack_config))

if args.dataset == 'cifar10':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100
elif args.dataset == 'tiny':
    args.num_classes = 200
elif args.dataset == 'svhn':
    args.num_classes = 10
elif args.dataset == 'mnist':
    args.num_classes = 10
else:
    args.num_classes = 10

nat_acc_list = []
adv_acc_list = []
def train_loop(epoch, net_helper):
    basic_net.train()
    for param_group in optimizer.param_groups:
        lr = lr_schedule(epoch)
        param_group['lr'] = lr
    global best_nat_acc
    global best_adv_acc
    global best_epoch
    global nat_acc_list
    global adv_acc_list
    index = 0
    nat_acc = 0
    adv_acc = 0
    loss_meter = AverageMeter()
    nat_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    nat_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, data in enumerate(iterator):
        inputs, targets, index = data
        if args.net_module.lower() == 'te' or args.net_module.lower() == 'ami':
            outputs_nat, outputs_adv, total_loss, nat_loss, adv_loss = net_helper.train(epoch, inputs, targets, optimizer)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_nat, outputs_adv, total_loss, nat_loss, adv_loss = net_helper.train(epoch, inputs, targets, index, optimizer)

        nat_acc = utils.get_acc(outputs_nat, targets)
        adv_acc = utils.get_acc(outputs_adv, targets)
        loss_meter.update(total_loss)
        nat_loss_meter.update(nat_loss)
        adv_loss_meter.update(adv_loss)
        nat_acc_meter.update(nat_acc)
        adv_acc_meter.update(adv_acc)

        if batch_idx % args.log_step == 0:
            msg = "\r| Step %3d, loss %.3f, n-loss %.3f, a-loss %.3f, nat acc %.1f, adv acc %.1f" % (batch_idx, total_loss, nat_loss, adv_loss, 100 * nat_acc, 100 * adv_acc)
            log.info(msg)

    writer.add_scalar(f'train/loss', loss_meter.avg, global_step = epoch)
    writer.add_scalar(f'train/nat_loss', nat_loss_meter.avg, global_step = epoch)
    writer.add_scalar(f'train/adv_loss', adv_loss_meter.avg, global_step = epoch)
    writer.add_scalar(f'train/nat_acc', nat_acc_meter.avg, global_step = epoch)
    writer.add_scalar(f'train/adv_acc', adv_acc_meter.avg, global_step = epoch)

    nat_acc = 0
    adv_acc = 0
    if (epoch > args.decay_epoch1-5) or (epoch % 10 == 0):
    # if epoch > -1:
        basic_net.eval()
        test_acc_meter = AverageMeter()
        test_adv_acc_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        test_adv_loss_meter = AverageMeter()

        num_classes = args.num_classes
        nat_cor_num = torch.zeros(num_classes).cuda()
        adv_cor_num = torch.zeros(num_classes).cuda()
        cls_num = torch.zeros(num_classes).cuda()
  
        for batch_idx, data in enumerate(testloader):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
        
            outputs_nat, nat_loss = net_helper.test(inputs, targets, adversary=None)
            outputs_adv, adv_loss = net_helper.test(inputs, targets, adversary=test_adversary)

            nat_acc = utils.get_acc(outputs_nat, targets)
            adv_acc = utils.get_acc(outputs_adv, targets)
            test_acc_meter.update(nat_acc)
            test_adv_acc_meter.update(adv_acc)
            test_loss_meter.update(nat_loss)
            test_adv_loss_meter.update(adv_loss)

        nat_acc = test_acc_meter.avg
        adv_acc = test_adv_acc_meter.avg

        writer.add_scalar(f'test/acc', 100.0*test_acc_meter.avg, global_step = epoch)
        writer.add_scalar(f'test/adv_acc', 100.0*test_adv_acc_meter.avg, global_step = epoch)
        writer.add_scalar(f'test/loss', test_loss_meter.avg, global_step = epoch)
        writer.add_scalar(f'test/adv_loss', test_adv_loss_meter.avg, global_step = epoch)
        msg = f'| Lr:{lr:.4f}. Test acc:{100.0*test_acc_meter.avg:.2f}, Adv acc:{100.0*test_adv_acc_meter.avg:.2f},Best acc:{100.0*best_nat_acc:.2f} Best adv acc:{100.0*best_adv_acc:.2f}, Best epoch:{best_epoch}'
        log.info(msg)

    if adv_acc > best_adv_acc:
        best_nat_acc = nat_acc
        best_adv_acc = adv_acc
        best_epoch = epoch
        msg = f'| Saving {args.net_module} {args.save_name} best epoch {epoch}\r'
        f_path = save_point + args.save_name+f'-best.t7'
    else:
        msg = f'| Saving {args.net_module} {args.save_name} latest epoch {epoch}\r'
        f_path = save_point + args.save_name+f'-latest.t7'
    if epoch == args.decay_epoch1:
        msg = f'| Saving {args.net_module} {args.save_name} epoch {epoch}\r'
        f_path = save_point + args.save_name+f'-{epoch}.t7'
    state = {
            'net': basic_net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_acc': best_nat_acc,
            'best_adv_acc': best_adv_acc,
            'best_epoch': best_epoch,
        }
    torch.save(state, f_path)
    if args.aux_net:
        aux_path = save_point + args.save_name+f'-aux.t7'
        aux_state = {
                'net': aux_net.state_dict(),
            }
        torch.save(aux_state, aux_path)
    log.info(msg)

log.info(f'==> Building network {args.net_type}..')
basic_net, net_type = get_Network(args)
basic_net =  basic_net.to(device)
log.info(f'==> Set lr schedule:{args.lr_schedule}')
lr_schedule = get_schedule(args)
log.info(f'==> Set optimaizer:{args.optim}')
optimizer = optim.SGD(basic_net.parameters(),lr=args.lr_max,momentum=args.momentum,weight_decay=args.weight_decay)
if args.swa:
    log.info(f'==> Set weight average : SWA')
    optimizer_swa = SWA(optimizer, swa_start=50, swa_freq=10, swa_lr=0.01)

if args.resume and args.init_model_pass != '-1':
    log.info('==> Resuming from checkpoint..')
    save_point = args.model_dir+args.dataset+os.sep
    f_path = save_point + args.save_name+f'-latest.t7'
    # f_path = save_point + f'cifar10-topo-v85-100.t7'

    if not os.path.isfile(f_path):
        log.info('==> No checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        basic_net.load_state_dict(checkpoint['net'])
        start_epoch  = checkpoint['epoch']
        best_nat_acc = checkpoint['best_acc']
        best_adv_acc = checkpoint['best_adv_acc']
        best_epoch   = checkpoint['best_epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info(f'==> from {f_path}')
        msg = f'==> epoch:{start_epoch}, best epoch:{best_epoch}, best acc:{best_nat_acc:.2f}'
        log.info(msg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    else:
        pass
else:
    log.info('==> Train from scratch...')
    start_epoch = 0
    best_nat_acc = 0
    best_adv_acc = 0
    best_epoch = 0

if args.aux_net:
    if args.zoo_model:
        print('==> Load model from model zoo !')
        print(f'==> Net name:{args.model_name}')
        from robustbench import load_model
        aux_net = load_model(model_name=args.model_name, dataset=args.dataset, norm=eval_attack_config['norm'])
        aux_net = aux_net.to(device)
    else:
        log.info(f'==> Building Aux net {args.aux_type}..')
        args.net_type = args.aux_type
        args.depth = args.aux_depth
        aux_net, aux_type = get_Network(args)
        aux_net = aux_net.to(device)
        aux_path = save_point + args.aux_name+f'-latest.t7'
        if os.path.isfile(aux_path):
            log.info('==> Load aux pretrained weight..')
            checkpoint = torch.load(aux_path)
            aux_net.load_state_dict(checkpoint['net'])
else:
    aux_net = basic_net.to(device)

if args.multi_gpu:
    basic_net = torch.nn.DataParallel(basic_net)
    aux_net = torch.nn.DataParallel(aux_net)

log.info(f'==> Building test adversary {args.attack_method.upper()}..')
test_adversary = get_adversary(basic_net, args.attack_method, eval_attack_config)

log.info(f'==> Building module {args.net_module.upper()}..')
net_helper = get_Module(basic_net, args, aux_net)

log.info(f'==> Preparing {args.dataset} data..')
if args.net_module == 'te':
    trainloader, testloader = get_te_dataloaders(train_batch=args.batch_size_train, test_batch=args.test_batch)
if args.net_module == 'ami':
    trainloader, testloader = net_helper.get_dataset(args.dataset)
else:
    trainloader, testloader = data_loader(dataset=args.dataset, train_batch=args.batch_size_train, test_batch=args.test_batch)

log.info(f'==> Training Stage ...')
start_time = time.time()
for epoch in range(start_epoch, args.max_epoch+1):
    train_loop(epoch, net_helper)
    duration = time.time() - start_time
    h,m,s=get_hms(duration)
    msg = '| Epoch: %d / %d, using time:%d:%02d:%02d \n' % (epoch,args.max_epoch,h,m,s)
    log.info(msg)
if args.swa:
    optimizer.swap_swa_sgd()
log.info('| ~~~~~~~~~~~~~ args ~~~~~~~~~~~~~~~ |')
log.info(str(args))
log.info('| ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |')
log.info(args.save_name+' Finished !')
