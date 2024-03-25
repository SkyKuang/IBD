import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from .wideresnet import *
from .LeNet import *
from .resnet_small import *
from .resnet_large import *
from .vgg import *
from .vgg_bnn import *
from .preactresnet import *
from .densenet import *
from .mobilenet import *
from .shufflenet import *
from .googlenet import *
from .model_utils import *

def get_Network(args):
    num_classes = args.num_classes
    if (args.net_type == 'wide'):
        net = WideResNet(depth=args.depth, dataset=args.dataset, widen_factor=args.widen_factor,num_classes=num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'res-l'):
        net = get_ResNet_large(args.depth, num_classes, dataset=args.dataset)
        file_name = f'resnet-large-{args.depth}'
    elif (args.net_type == 'res-s'):
        net = get_ResNet_small(args.depth, num_classes, dataset=args.dataset)
        file_name = f'resnet-small-{args.depth}'
    elif (args.net_type == 'pre-res'):
        net = get_PreActResNet(args.depth, num_classes, dataset=args.dataset)
        file_name = f'PreAct-ResNet-{args.depth}'
    elif (args.net_type == 'vgg'):
        net = get_VGG(args.depth, num_classes, dataset=args.dataset)
        file_name = f'vgg-{args.depth}'
    elif (args.net_type == 'vgg-bnn'):
        net = get_VGG_BNN(args.depth, num_classes, dataset=args.dataset)
        file_name = f'vgg-bnn-{args.depth}'
    elif (args.net_type == 'dense'):
        net = get_Densenet(args.depth, num_classes, dataset=args.dataset)
        file_name = f'densenet-{args.depth}'
    elif (args.net_type == 'mobi'):
        net = get_Mobilenet(num_classes, dataset=args.dataset)
        file_name = f'mobile-net'
    elif (args.net_type == 'google'):
        net = get_GoogLeNet(num_classes, dataset=args.dataset)
        file_name = f'google-net'
    elif (args.net_type == 'shuffle'):
        net = get_ShuffleNet(num_classes, dataset=args.dataset)
        file_name = f'shuffle-net'
    elif (args.net_type == 'lenet'):
        net = LeNet5()
        file_name = 'lenet'
    else:
        raise NotImplementedError('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
    return net, file_name