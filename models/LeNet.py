from collections import OrderedDict
import torch.nn as nn
import pdb

class LeNet5(nn.Module):
    def __init__(self, drop=0.5):
        super(LeNet5, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('BN1', nn.BatchNorm2d(32)),
            ('relu1', activ),
            ('maxpool0', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(32, 64, 3)),
            ('BN2', nn.BatchNorm2d(64)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(64, 128, 3)),
            ('BN3', nn.BatchNorm2d(128)),
            ('relu3', activ),
            # ('conv4', nn.Conv2d(128, 256, 3)),
            # ('BN4', nn.BatchNorm2d(256)),
            # ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 1 * 1, 81)),
            ('BN5', nn.BatchNorm1d(81)),
            ('relu1', activ),
        ]))

        self.fc = nn.Linear(81, self.num_labels)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        features = features.view(-1, 128 * 1 * 1)
        features = self.classifier(features)
        logits = self.fc(features)
        return logits