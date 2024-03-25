import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
from utils import one_hot_tensor

def EOT_PGD_attack(model, inputs, targets, eot_iter=10, num_steps=10, epsilon=0.03137, step_size=0.0078):
    model.eval()
    x = inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
    grad = torch.zeros_like(x)
    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            for _ in range(eot_iter):
                logits = model(x)
                loss_indiv = F.cross_entropy(logits, targets)
                loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x])[0].detach()    
            grad /= float(eot_iter)
        x = x.detach() + step_size*torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return x.detach()

def PGD_attack(net, inputs, targets, num_steps=10, epsilon=0.03137, step_size=0.0078):
    basic_net = net
    x = inputs.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    basic_net.eval()

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logit = basic_net(x)
            loss = F.cross_entropy(logit, targets, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size*torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    return x.detach()

def TRADES_attack(model,x_natural, perturb_steps=10, epsilon=0.03137, step_size=0.0078):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    logits_nat = model(x_natural)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_adv = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                    F.softmax(logits_nat, dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv

class CWLoss(nn.Module):
    def __init__(self, num_classes=10, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss