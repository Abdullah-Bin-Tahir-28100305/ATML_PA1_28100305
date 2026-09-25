

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()                     
        self.net = net
        self.feature_dim = net.fc.in_features           

    def _stem(self, x):
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))                   
        x = n.maxpool(x)                                
        return x

    def forward_features(self, x):
        n = self.net
        x = self._stem(x)
        x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        x = n.avgpool(x)                                
        return torch.flatten(x, 1)                      

    def forward(self, x):
        feats = self.forward_features(x)
        return self.net.fc(feats)

    def forward_from_features(self, feats):
        return self.net.fc(feats)

    def phi_pre(self, x):
        n = self.net
        x = self._stem(x)
        x = n.layer1(x)
        x = n.layer2(x)                                 
        return x

    def phi_post(self, h):
       
        n = self.net
        x = n.layer3(h)                                 
        x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)                      

    def forward_split_logits(self, h):
        return self.net.fc(self.phi_post(h))
