'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import Normalize
import numpy as np

def gaussian_act(x, u=1.5, d=1):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 标准差

    返回:
    p -- 高斯分布值
    """
    d_2 = d * d * 2
    num_exp = -(torch.square(x - u) / d_2)
    exp = torch.exp(num_exp)
    pi = np.pi
    alpha = 1 / (np.sqrt(2 * pi) * d)
    p = alpha * exp
    return p

def gaussian_grad(x,u=1.5,d=1):
    d_2 = d*d*2
    g1 = torch.exp(x)
    g2 = -2*(x-u)/d_2
    pi = np.pi
    alpha = 1 / (np.sqrt(2 * pi) * d)
    grad = alpha*g1*g2
    return grad

class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)               # relu就是截断负数，让所有负数等于0
        output = gaussian_act(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu    
        input_, = ctx.saved_tensors # 把上面保存的input_输出
        # if ctx.needs_input_grad[0]:# 判断self.saved_tensors中的Variable是否需要进行反向求导计算梯度
        #     print('input_ need grad')
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0       # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        grad_input = grad_input*gaussian_grad(input_)
        return grad_input

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))


        out += shortcut
        return out


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (PreActBlock, [2,2,2,2]),
        '34': (PreActBlock, [3,4,6,3]),
        '50': (PreActBottleneck, [3,4,6,3]),
        '101':(PreActBottleneck, [3,4,23,3]),
        '152':(PreActBottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class PreActResNet(nn.Module):
    def __init__(self,depth, num_classes=10, dataset='cifar10'):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.norm1_layer = Normalize(dataset)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_feat(self, x):
        out = self.norm1_layer(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        feat = out.view(out.size(0), -1)
        return feat

    def forward(self,x):
        feat = self.extract_feat(x)
        out = self.linear(feat)
        return out

def get_PreActResNet(depth, num_classes=10, dataset='cifar10'):
    return PreActResNet(depth, num_classes, dataset)

def test():
    net = PreActResNet()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
