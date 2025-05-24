import torch
from torch import nn
import torch.nn.functional as F


class WCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Pred: (batch_size, num_classes, height, width)
        # Target: (batch_size, num_classes, height, width) [one-hot encoded]

        # Calculate the weight for each class based on the target's class distribution
        num_classes = pred.size(1)
        weight = torch.zeros(num_classes, device=pred.device)

        for i in range(num_classes):
            class_target = target[:, i, :, :]
            neg = (1. - class_target).sum()  # Number of 0s for class i
            pos = class_target.sum()  # Number of 1s for class i
            beta = neg / (neg + pos)  # Proportion of 0s for class i
            weight[i] = beta  # Store the class-specific weight

        # Now, we need to calculate weighted binary cross entropy for each class
        loss = 0.0
        for i in range(num_classes):
            # For each class, compute binary cross entropy loss
            pred_class = pred[:, i, :, :]
            target_class = target[:, i, :, :]
            class_weight = weight[i]  # Weight for the current class

            # Weight is applied based on the target (0 or 1), so we can compute it in the following way
            class_loss = F.binary_cross_entropy(pred_class, target_class, reduction='none')
            class_loss = class_loss * (target_class * class_weight + (1. - target_class) * (1. - class_weight))

            loss += class_loss.sum()

        # Apply the reduction (mean or sum)
        if self.reduction == 'mean':
            loss = loss / target.numel()  # Normalize by the total number of elements
        return loss
