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

class Attack_Base():
    # Back-propogate
    def __init__(self, basic_net):
        super(Attack_Base, self).__init__()
        self.basic_net = basic_net

    def attack(self, inputs, targets):
        return inputs