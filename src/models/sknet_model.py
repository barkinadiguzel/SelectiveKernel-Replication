import torch
import torch.nn as nn
from blocks.sk_blocks import SKUnit
from config import SKConfig


class SKNet(nn.Module):
    def __init__(self, config: SKConfig, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = config.in_channels
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, config.num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(SKUnit(self.inplanes, planes, stride, self.config))
        self.inplanes = planes * 4

        for _ in range(1, blocks):
            layers.append(SKUnit(self.inplanes, planes, 1, self.config))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
