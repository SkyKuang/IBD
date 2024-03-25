from __future__ import print_function
from __future__ import absolute_import
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import pdb
import datetime
import numpy as np
from tqdm import tqdm
import utils
from models import get_Network
from defense import get_Module
from attacks import get_adversary
from config import args_cf, attack_config
from dataloader import data_loader
from utils import get_hms, AverageMeter, logger, load_checkpoint
from advertorch.bpda import BPDAWrapper

torch.manual_seed(111)
torch.cuda.manual_seed(111)
np.random.seed(111)
torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

def eval_attack(net_helper, adversary):
    basic_net.eval()
    aux_net.eval()
    feat_list = []
    label_list = []
    acc_meter = AverageMeter()
    iterator = tqdm(testloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = net_helper.test(inputs, targets, adversary)
        acc = utils.get_acc(outputs, targets)
        acc_meter.update(acc)
        if batch_idx % args.log_step == 0:
            print(f'| Step:{batch_idx}, acc:{100*acc:.2f}, avg_acc:{100*acc_meter.avg:.2f}')

    acc = 100*acc_meter.avg
    return acc

print('==> Building args..')
args = args_cf.get_eval_args()
save_point = args.model_dir+args.dataset+os.sep
log = logger(path=save_point, file_name=args.save_name)
log.info(str(args))
log.info(str(attack_config))

if args.dataset == 'cifar10':
    args.num_classes = 10
    attack_config['num_classes'] = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100
    attack_config['num_classes'] = 100
elif args.dataset == 'tiny':
    args.num_classes = 200
    attack_config['num_classes'] = 200
elif args.dataset == 'svhn':
    args.num_classes = 10
    attack_config['num_classes'] = 10
elif args.dataset == 'mnist':
    args.num_classes = 10
    attack_config['num_classes'] = 10
else:
    args.num_classes = 10
    attack_config['num_classes'] = 10

print(f'==> Preparing {args.dataset} data..')
trainloader, testloader = data_loader(dataset=args.dataset, test_batch=args.test_batch)

print('==> Building network..')
if args.zoo_model:
    print('==> Load model from model zoo !')
    print(f'==> Net name:{args.model_name}')
    from robustbench import load_model
    basic_net = load_model(model_name=args.model_name, dataset=args.dataset, norm=attack_config['norm'])
    basic_net = basic_net.to(device)
    basic_net.eval()
else:
    basic_net, net_name = get_Network(args)
    print(f'==> Net name:{net_name}')
    basic_net = basic_net.to(device)
    if args.multi_gpu:
        basic_net = torch.nn.DataParallel(basic_net)

    if args.init_model_pass == 'best':
        f_path = save_point + args.save_name+f'-best.t7'
        print('==> Load best checkpoint..')
    elif args.init_model_pass == 'latest':
        print('==> Load latest checkpoint..')
        f_path = save_point + args.save_name+f'-latest.t7'
    else:
        f_path = save_point + args.save_name+f'.t7'
    checkpoint = torch.load(f_path)
    basic_net.load_state_dict(checkpoint['net'])
    basic_net.eval()
    epoch = checkpoint['epoch']
    best_epoch = checkpoint['best_epoch']
    acc = checkpoint['best_acc']
    adv_acc = checkpoint['best_adv_acc']
    log.info(f'==> acc:{acc*100:.2f}, adv_acc:{adv_acc*100:.2f}, epoch:{epoch}, best_epoch:{best_epoch}')

if args.aux_net or args.black_attack:
    print('==> Building Aux net..')
    args.net_type = args.aux_type
    args.depth = args.aux_depth
    aux_net, aux_name = get_Network(args)
    aux_net = aux_net.to(device)
    if args.multi_gpu:
        aux_net = torch.nn.DataParallel(aux_net)
    print(f'==> Aux name:{aux_name}')
    aux_path = save_point + args.aux_name + f'-best.t7'
    # aux_path = save_point + args.aux_name + f'-aux.t7'
    if os.path.isfile(aux_path):
        if args.multi_gpu:
            aux_net = torch.nn.DataParallel(aux_net)
        checkpoint = torch.load(aux_path)
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        acc = checkpoint['best_acc']
        adv_acc = checkpoint['best_adv_acc']
        aux_net.load_state_dict(checkpoint['net'])
        print(f'==> Load checkpoint success, acc:{acc*100:.2f}, adv_acc:{adv_acc*100:.2f}, epoch:{epoch}, best_epoch:{best_epoch}')
else:
    aux_net = basic_net

if args.bpda:
    print('==> BPDA Attack !')
    if args.bpda_type.upper() == 'AEP':
        defense_layer = AEP(aux_net, args)
    elif args.bpda_type.upper() == 'ME':
        defense_layer = ME(args)
    else:
        raise NotImplementedError("Please implement the net first!")
    defense_withbpda = BPDAWrapper(defense_layer, forwardsub=lambda x: x)
    attack_net = nn.Sequential(defense_withbpda, basic_net)
elif args.black_attack:
    print('==> Black-box transform Attack !')
    attack_net = aux_net
else:
    attack_net = basic_net

print('==> Building module..')
net_helper = get_Module(basic_net, args, aux_net=aux_net)
log.info(f'==> Module:{args.net_module.upper()}')
log.info(f'==> Model Name :{args.save_name}')

if args.net_module == 'VIS':
    net_helper.vis_feature(train_loader=trainloader, test_loader=testloader)
    pdb.set_trace()
    exit(0)

if args.benchmark:
    if args.zoo_model:
        log_str = args.model_name
    else:
        log_str = args.save_name
    print('==> Robust Benchmark testing..')
    from robustbench.eval import benchmark
    basic_net.eval()
    device = torch.device("cuda:0")
    clean_acc, robust_acc = benchmark(basic_net,
                                    dataset='cifar10',
                                    data_dir='/home/khf/datasets/cifar10',
                                    threat_model='Linf',
                                    eps=8.0/255,
                                    batch_size=100,
                                    log=f'log/{log_str}.txt',
                                    device=device)

    log.info(f'==> {log_str}')
    log.info(f'==> Benchmark Model claen acc :{clean_acc}')
    log.info(f'==> Benchmark Model rboust acc :{robust_acc}')
    exit(0)

attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)
for attack_idx in range(attack_num):
    attack_method = attack_list[attack_idx]
    print(f'==> {attack_method} attack..')
    adversary = get_adversary(attack_net, attack_method, attack_config)
    start_time = time.time()
    acc = eval_attack(net_helper, adversary)
    duration = time.time() - start_time
    h,m,s = get_hms(duration)
    msg = '==> %s eval:%.2f, using time:%d:%02d:%02d \n' % (attack_method, acc, h, m, s)
    log.info(msg)

log.info(f'==> {args.save_name} test finished!')
