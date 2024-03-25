import torch
import torch.nn as nn
from torch.autograd import Variable
from model_utils import Normalize

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, depth=13, num_classes=10, dataset='cifar10'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[str(depth)])
        self.classifier = nn.Linear(512, num_classes)
        self.norm1_layer = Normalize(dataset)

    def forward(self, x):
        x = self.norm1_layer(x)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def get_VGG(depth, num_classes, dataset='cifar10'):
    return VGG(depth, num_classes, dataset)

if __name__ == "__main__":
    net = VGG(16, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
