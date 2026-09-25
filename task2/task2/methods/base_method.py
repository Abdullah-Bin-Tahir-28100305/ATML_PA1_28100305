

import torch.nn as nn


class BaseMethod(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()                # standard multiclass loss

    def classification_loss(self, source_logits, source_labels):

        return self.ce(source_logits, source_labels)

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError                      # each method overrides this

    def extra_modules(self):
      
        return []
