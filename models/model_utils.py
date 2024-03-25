import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn import init
from torch.autograd import Variable
import sys
import pdb
import random
import numpy as np
import math
from torch.autograd import Function
from torch.nn import Parameter


device = 'cuda' if torch.cuda.is_available() else 'cpu'

meanstd_dataset = {
    # 'svhn': [[0.0, 0.0, 0.0],
                # [1.0, 1.0, 1.0]],
    'cifar10': [[0.49139968, 0.48215827, 0.44653124],
                [0.24703233, 0.24348505, 0.26158768]],
    'cifar100': [[0.49139968, 0.48215827, 0.44653124],
                [0.24703233, 0.24348505, 0.26158768]],
    'mnist': [[0.13066051707548254],
                [0.30810780244715075]],
    'fashionmnist': [[0.28604063146254594],
                [0.35302426207299326]],
    'imagenet': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'tiny': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'stl': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'svhn': [[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]],
}

imgsize_dataset = {
    'cifar10':32,
    'cifar100':32,
    'svhn':32,
    'mnist':28,
    'tiny':64,
    'imagenet':224,
}

class Normalize(nn.Module):
    def __init__(self, dataset='cifar10', input_channels=3):
        super(Normalize, self).__init__()
        self.input_size = imgsize_dataset[dataset.lower()]
        self.input_channels = input_channels

        self.mean, self.std = meanstd_dataset[dataset.lower()]
        self.mean = torch.Tensor(np.array(self.mean)[:, np.newaxis, np.newaxis]).cuda()
        self.mean = self.mean.expand(self.input_channels, self.input_size, self.input_size).cuda()
        self.std = torch.Tensor(np.array(self.std)[:, np.newaxis, np.newaxis]).cuda()
        self.std = self.std.expand(self.input_channels, self.input_size, self.input_size).cuda()

    def forward(self, input):
        device = input.device
        output = input.sub(self.mean.to(device)).div(self.std.to(device))
        return output

class Proj_head(nn.Module):
    def __init__(self, ch, bn_names=None):
        super(Proj_head, self).__init__()
        self.in_features = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)
        self.fc2 = nn.Linear(ch, ch//4, bias=False)
        self.bn2 = batch_norm_multiple(nn.BatchNorm1d, ch//4, bn_names)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        # debug
        # print("adv attack: {}".format(flag_adv))

        x = self.fc1(x)
        x = self.bn1([x, bn_name])

        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2([x, bn_name])

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = F.dropout(x,p=0.5)
            x = self.layer2(x)
            x = F.dropout(x,p=0.3)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = F.dropout(x,p=0.3)
            x = self.layer3(x)
        else:
            raise Exception
        return x 
        
class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}
        return

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out

# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/model_ops.py

def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.Linear)):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                print('Init style not recognized...')
        elif isinstance(module, nn.Embedding):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        else:
            pass

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def embedding(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, num_classes, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)

        if spectral_norm:
            self.embed0 = sn_embedding(num_classes, num_features)
            self.embed1 = sn_embedding(num_classes, num_features)
        else:
            self.embed0 = embedding(num_classes, num_features)
            self.embed1 = embedding(num_classes, num_features)

    def forward(self, x, y):
        gain = (1 + self.embed0(y)).view(-1, self.num_features, 1, 1)
        bias = self.embed1(y).view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias

class ConditionalBatchNorm2d_for_skip_and_shared(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, z_dims_after_concat, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)

        if spectral_norm:
            self.gain = snlinear(z_dims_after_concat, num_features, bias=False)
            self.bias = snlinear(z_dims_after_concat, num_features, bias=False)
        else:
            self.gain = linear(z_dims_after_concat, num_features, bias=False)
            self.bias = linear(z_dims_after_concat, num_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias

class Self_Attn(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_channels, spectral_norm):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels

        if spectral_norm:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma*attn_g

def make_mask(labels, n_cls):
    device = labels.device
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    mask_multi = np.zeros([n_cls, n_samples])
    for c in range(n_cls):
        c_indices = np.where(labels==c)
        mask_multi[c, c_indices] =+1

    mask_multi = torch.tensor(mask_multi).type(torch.long)
    return mask_multi.to(device)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.net(x)
        return out

class DisNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DisNet, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        if self.std > 0:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.data.resize_(x.size()).normal_(0, self.std)
            return x + self.buffer
        return x

class NoiseFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N):
        eps.normal_()
        ctx.save_for_backward(mu, sigma, eps)
        ctx.sigma_0 = sigma_0
        ctx.N = N
        return mu + torch.exp(sigma) * eps

    @staticmethod
    def backward(ctx, grad_output):
        mu, sigma, eps = ctx.saved_tensors
        sigma_0, N = ctx.sigma_0, ctx.N
        grad_mu = grad_sigma = grad_eps = grad_sigma_0 = grad_N = None
        tmp = torch.exp(sigma)
        if ctx.needs_input_grad[0]:
            grad_mu = grad_output + mu/(sigma_0*sigma_0*N)
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_output*tmp*eps - 1 / N + tmp*tmp/(sigma_0*sigma_0*N)
        return grad_mu, grad_sigma, grad_eps, grad_sigma_0, grad_N

class IdFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N):
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

noise_fn = NoiseFn.apply


class RandLinear(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_features, out_features, bias=True):
        super(RandLinear, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.init_s = init_s
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.sigma_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_buffer('eps_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        self.eps_weight.data.zero_()
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)
            self.eps_bias.data.zero_()

    def forward_(self, input):
        weight = noise_fn(self.mu_weight, self.sigma_weight, self.eps_weight, self.sigma_0, self.N)
        bias = None
        if self.mu_bias is not None:
            bias = noise_fn(self.mu_bias, self.sigma_bias, self.eps_bias, self.sigma_0, self.N)
        return F.linear(input, weight, bias)

    def forward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_weight = math.log(self.sigma_0) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.sigma_0 ** 2) - 0.5
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()
            kl_bias = math.log(self.sigma_0) - self.sigma_bias + (sig_bias**2 + self.mu_bias**2) / (2 * self.sigma_0 ** 2) - 0.5
        out = F.linear(input, weight, bias)
        kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
        return out, kl

class RandConv2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RandConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_parameter('eps_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def forward_(self, input):
        weight = noise_fn(self.mu_weight, self.sigma_weight, self.eps_weight, self.sigma_0, self.N)
        bias = None
        if self.mu_bias is not None:
            bias = noise_fn(self.mu_bias, self.sigma_bias, self.eps_bias, self.sigma_0, self.N)
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def forward(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_weight = math.log(self.sigma_0) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.sigma_0 ** 2) - 0.5
        bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()
            kl_bias = math.log(self.sigma_0) - self.sigma_bias + (sig_bias**2 + self.mu_bias**2) / (2 * self.sigma_0 ** 2) - 0.5
        out = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_weight.sum() + kl_bias.sum() if self.mu_bias is not None else kl_weight.sum()
        return out, kl

class RandBatchNorm2d(nn.Module):
    def __init__(self, sigma_0, N, init_s, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(RandBatchNorm2d, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.num_features = num_features
        self.init_s = init_s
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.mu_weight = Parameter(torch.Tensor(num_features))
            self.sigma_weight = Parameter(torch.Tensor(num_features))
            self.register_buffer('eps_weight', torch.Tensor(num_features))
            self.mu_bias = Parameter(torch.Tensor(num_features))
            self.sigma_bias = Parameter(torch.Tensor(num_features))
            self.register_buffer('eps_bias', torch.Tensor(num_features))
        else:
            self.register_parameter('mu_weight', None)
            self.register_parameter('sigma_weight', None)
            self.register_buffer('eps_weight', None)
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
            self.register_buffer('eps_bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.mu_weight.data.uniform_()
            self.sigma_weight.data.fill_(self.init_s)
            self.mu_bias.data.zero_()
            self.sigma_bias.data.fill_(self.init_s)
            self.eps_weight.data.zero_()
            self.eps_bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward_(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        # generate weight and bias
        weight = bias = None
        if self.affine:
            weight = noise_fn(self.mu_weight, self.sigma_weight, self.eps_weight, self.sigma_0, self.N)
            bias = noise_fn(self.mu_bias, self.sigma_bias,  self.eps_bias, self.sigma_0, self.N)

        return F.batch_norm(input, self.running_mean, self.running_var, weight, bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        # generate weight and bias
        weight = bias = None
        if self.affine:
            sig_weight = torch.exp(self.sigma_weight)
            weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
            kl_weight = math.log(self.sigma_0) - self.sigma_weight + (sig_weight**2 + self.mu_weight**2) / (2 * self.sigma_0 ** 2) - 0.5
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()
            kl_bias = math.log(self.sigma_0) - self.sigma_bias + (sig_bias**2 + self.mu_bias**2) / (2 * self.sigma_0 ** 2) - 0.5

        out = F.batch_norm(input, self.running_mean, self.running_var, weight, bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)
        kl = kl_weight.sum() + kl_bias.sum()
        return out, kl
