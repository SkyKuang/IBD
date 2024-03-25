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
import torchvision.models as models
import eagerpy as ep
import foolbox.attacks as fa
import foolbox
from foolbox import PyTorchModel, accuracy, samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""A powerful adversarial attack that requires neither gradients
nor probabilities [#Chen19].
Args:
    init_attack : Attack to use to find a starting points. Defaults to
        LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
    steps : Number of optimization steps within each binary search step.
    initial_gradient_eval_steps: Initial number of evaluations for gradient estimation.
        Larger initial_num_evals increases time efficiency, but
        may decrease query efficiency.
    max_gradient_eval_steps : Maximum number of evaluations for gradient estimation.
    stepsize_search : How to search for stepsize; choices are 'geometric_progression',
        'grid_search'. 'geometric progression' initializes the stepsize
        by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
        until reaching the target side of the boundary. 'grid_search'
        chooses the optimal epsilon over a grid, in the scale of
        ||x_t - x||_p.
    gamma : The binary search threshold theta is gamma / d^1.5 for
                l2 attack and gamma / d^2 for linf attack.
    tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
        summaries will be disabled (default). If None, the logdir will be
        runs/CURRENT_DATETIME_HOSTNAME.
    constraint : Norm to minimize, either "l2" or "linf"
References:
    .. [#Chen19] Jianbo Chen, Michael I. Jordan, Martin J. Wainwright,
    "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack",
    https://arxiv.org/abs/1904.02144
"""

class Attack_HSJ():
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_HSJ, self).__init__()
        self.basic_net = basic_net
        self.epsilon = config['epsilon']
        self.clip_min = config['clip_min']
        self.clip_max = config['clip_max']
        self.targeted = config['targeted']
        self.num_classes = config['num_classes']
        self.num_steps = config['num_steps']
        self.basic_net.eval()

        self.fmodel = PyTorchModel(self.basic_net, bounds=(self.clip_min, self.clip_max))
        self.adversary = fa.HopSkipJump()

    def attack(self, inputs, targets):
        raw_advs, clipped_advs, success = self.adversary(self.fmodel, inputs, targets, epsilons=0.1)
        return clipped_advs


