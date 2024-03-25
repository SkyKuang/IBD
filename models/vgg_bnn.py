'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .model_utils import RandBatchNorm2d
from .model_utils import RandConv2d
from .model_utils import RandLinear
from .model_utils import Normalize

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG_BNN(nn.Module):
    def __init__(self, sigma_0=0.08, N=50000, init_s=0.08, depth=16, num_classes=10, dataset='cifar10', img_width=32):
        super(VGG_BNN, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.img_width = img_width
        self.classifier = RandLinear(sigma_0, N, init_s, 512, num_classes)
        self.features = self._make_layers(cfg[str(depth)])
        self.norm1_layer = Normalize(dataset)

    def forward(self, x, flg=False):
        kl_sum = 0
        out = x
        for l in self.features:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        out, kl = self.classifier.forward(out)
        kl_sum += kl
        if flg:
            return out, kl
        else:
            return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [
                        RandConv2d(self.sigma_0, self.N, self.init_s, in_channels, x, kernel_size=3, padding=1),
                        RandBatchNorm2d(self.sigma_0, self.N, self.init_s, x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

def get_VGG_BNN(depth, num_classes, dataset='cifar10'):
    return VGG_BNN(depth=depth, num_classes=num_classes, dataset=dataset)