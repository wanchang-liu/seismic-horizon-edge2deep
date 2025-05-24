from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, reduction='mean'):
        """
        :param alpha: Class weight for each class, can be a tensor for each class.
        :param gamma: Modulating factor to reduce the relative loss for well-classified examples.
        :param reduction: Reduction method ('none' | 'mean' | 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: (batch_size, num_classes, height, width)
        # target: (batch_size, num_classes, height, width) [one-hot encoded]

        # Calculate the focal loss for each class
        num_classes = pred.size(1)
        loss = 0.0

        for i in range(num_classes):
            # For each class, calculate the focal loss
            pred_class = pred[:, i, :, :]
            target_class = target[:, i, :, :]

            # Compute the binary cross-entropy for this class
            bce_loss = F.binary_cross_entropy(pred_class, target_class, reduction='none')

            # Calculate p_t
            p_t = target_class * pred_class + (1. - target_class) * (1. - pred_class)

            # Calculate the modulating factor (1 - p_t)^gamma
            modulating_factor = (1. - p_t) ** self.gamma

            # Calculate the alpha weighting factor
            alpha_factor = target_class * self.alpha + (1. - target_class) * (1. - self.alpha)

            # Focal loss for the current class
            focal_loss = alpha_factor * modulating_factor * bce_loss
            loss += focal_loss.sum()  # Sum up the loss for this class

        # Apply the reduction (mean or sum)
        if self.reduction == 'mean':
            loss = loss / target.numel()  # Normalize by the total number of elements
        elif self.reduction == 'sum':
            pass  # No change if reduction is sum
        return loss
