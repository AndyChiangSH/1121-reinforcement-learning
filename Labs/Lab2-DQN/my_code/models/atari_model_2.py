import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(10, 10), stride=(4, 3)),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=(10, 10), stride=(2, 2)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(10, 10), stride=(1, 1)),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # My
        # print("x.shape:", x.shape)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        # print("x.shape after cnn:", x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

