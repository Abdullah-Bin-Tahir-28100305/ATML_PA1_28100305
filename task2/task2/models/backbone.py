
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = net.fc.in_features          # 512 for ResNet-18
        
        net.fc = nn.Identity()
        self.net = net
        

    def forward(self, x):
        return self.net(x)                             # (N, 512)

    def set_bn_eval(self):
        
        for m in self.net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()                               # freeze running stats
                
