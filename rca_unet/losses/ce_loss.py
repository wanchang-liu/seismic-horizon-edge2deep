import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Assuming the target is one-hot encoded, we need to convert it to class indices
        target = torch.argmax(target, dim=1)  # Convert one-hot to class indices

        # Calculate cross-entropy loss
        loss = F.cross_entropy(pred, target, reduction=self.reduction)
        return loss
