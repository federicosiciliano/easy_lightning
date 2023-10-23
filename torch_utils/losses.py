import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

#https://github.com/AlanChou/Truncated-Loss
class GCELoss(nn.Module):

    def __init__(self, q=0.7, k=0.5):
        super(GCELoss, self).__init__()
        self.q = q
        self.k = k
        self.weights = {}
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        
        # Initialize weights for new indexes
        for index in indexes:
            if index.item() not in self.weights:
                self.weights[index.item()] = torch.nn.Parameter(data=torch.ones(1), requires_grad=False)
        
        loss = ((1-(Yg**self.q))/self.q)*torch.stack([self.weights[i.item()] for i in indexes]) - \
               ((1-(self.k**self.q))/self.q)*torch.stack([self.weights[i.item()] for i in indexes])
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = ((1-(self.k**self.q))/self.q) * torch.ones_like(targets, dtype=torch.float)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        for i, index in enumerate(indexes):
            self.weights[index.item()] = condition[i].float().to(logits.device)

