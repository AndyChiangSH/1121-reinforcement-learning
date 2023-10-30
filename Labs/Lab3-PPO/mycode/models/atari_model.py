import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AtariNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNet, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
        )
        self.action_logits = nn.Sequential(nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        self.value = nn.Sequential(nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, a=[]):
        #print(x.shape)
        x = x.float() / 255.
        x = self.cnn(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        value = self.value(x)
        value = torch.squeeze(value)

        logits = self.action_logits(x)
        
        dist = Categorical(logits=logits)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        if eval:
            action = torch.argmax(logits, axis=1)
        else:
            action = dist.sample()

        if len(a) == 0:
            action_log_prob = dist.log_prob(action)
        else:
            action_log_prob = dist.log_prob(a)
        #action_log_prob = dist.log_prob(action)
        
        dist_entropy = dist.entropy().mean()
        action_log_prob = torch.squeeze(action_log_prob)
        return action, action_log_prob, value, dist_entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                


