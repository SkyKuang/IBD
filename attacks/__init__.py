from.attack_utils import *
from .Base import *
from .AA import *
from .PGD import *
from .CW import *
from .FGSM import *
from .SPA import *
from .MIM import *
from .JSMA import *
from .BPDA import *
from .SPSA import *
from .BA import *
from .HSJ import *
from .GenA import *

def get_adversary(attack_net, attack_method, attack_config):
    if attack_method == 'natural':
        adversary = Attack_Base(attack_net)
    elif attack_method.upper() == 'FGSM':
        adversary = Attack_FGSM(attack_net, attack_config)
    elif attack_method.upper() == 'PGD':
        adversary = Attack_PGD(attack_net, attack_config)
    elif attack_method.upper() == 'CW':
        adversary = PGD_CW(attack_net, attack_config)
    elif attack_method.upper() == 'CW2':
        adversary = Attack_CW(attack_net, attack_config)
    elif attack_method.upper() == 'JSMA':
        adversary = Attack_JSMA(attack_net, attack_config)
    elif attack_method.upper() == 'MIM':
        adversary = Attack_MIM(attack_net, attack_config)
    elif attack_method.upper() == 'SPA':
        adversary = Attack_SPA(attack_net, attack_config)
    elif attack_method.upper() == 'SPSA':
        adversary = Attack_SPSA(attack_net, attack_config)
    elif attack_method.upper() == 'AUTO':
        adversary = Attack_Auto(attack_net, attack_config)
    elif attack_method.upper() == 'BA':
        adversary = Attack_BA(attack_net, attack_config)
    elif attack_method.upper() == 'HSJ':
        adversary = Attack_HSJ(attack_net, attack_config)
    elif attack_method.upper() == 'GEN':
        adversary = Attack_GenA(attack_net, attack_config)
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(attack_method))

    return adversary