from torch import nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, epsilon=1e-5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: (batch_size, num_classes, height, width)
        # target: (batch_size, num_classes, height, width) [one-hot encoded]

        num_classes = pred.size(1)
        total_loss = 0.0

        for i in range(num_classes):
            pred_class = pred[:, i, :, :]
            target_class = target[:, i, :, :]

            # Dice Loss for this class
            intersection = (pred_class * target_class).sum()  # Intersection
            total = (pred_class + target_class).sum()  # Total (union)
            dice_loss = 1. - (2. * intersection + self.epsilon) / (total + self.epsilon)

            # Focal Loss for this class
            bce_loss = F.binary_cross_entropy(pred_class, target_class, reduction='none')  # BCE Loss for each class
            p_t = target_class * pred_class + (1. - target_class) * (1. - pred_class)  # p_t for this class
            modulating_factor = (1. - p_t) ** self.gamma  # (1 − p_t)^gamma
            alpha_factor = target_class * self.alpha + (1. - target_class) * (1. - self.alpha)  # Alpha weight factor
            focal_loss = alpha_factor * modulating_factor * bce_loss

            # Combine Dice and Focal Loss
            total_loss += focal_loss.sum() + dice_loss

        # Apply the reduction ('mean', 'sum', or 'none')
        if self.reduction == 'mean':
            total_loss = total_loss / target.numel()  # Normalize by the total number of elements
        elif self.reduction == 'sum':
            pass  # No change if reduction is sum

        return total_loss
