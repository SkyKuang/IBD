import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from .defense_utils import *
from .AT_module import *
from .TRADES_module import *
from .TOPO_module import *

def get_Module(basic_net, args, aux_net=None):
    module_list = ['topo', 'trades', 'at']
    if args.net_module.lower() in module_list:
        if args.net_module.lower() == 'at':
            net_helper = AT_Module(basic_net, args,aux_net=aux_net)
        elif args.net_module.lower() == 'trades':
            net_helper = TRADES_Module(basic_net, args, aux_net=aux_net)
        elif args.net_module.lower() == 'ibd':
            net_helper = TOPO_Module(basic_net, args, aux_net=aux_net)
        else:
            pass
    else:
        raise NotImplementedError("Please implement the module first!")
    return net_helper