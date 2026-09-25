

import torch.nn as nn


class ClassifierHead(nn.Module):

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats):
        return self.fc(feats)                          # (N, num_classes) logits
