import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Pred: (batch_size, num_classes, height, width)
        # Target: (batch_size, num_classes, height, width) [one-hot encoded]

        # Initialize a list to store the Dice scores for each class
        dice_scores = []

        # Loop through each class
        for i in range(pred.size(1)):  # pred.size(1) is the number of classes
            pred_class = pred[:, i, :, :]
            target_class = target[:, i, :, :]

            intersection = (pred_class * target_class).sum()  # Intersection
            total = (pred_class + target_class).sum()  # Total (union)

            dice_score = (2. * intersection + self.epsilon) / (total + self.epsilon)
            dice_scores.append(dice_score)

        # Mean Dice loss over all classes
        loss = 1. - torch.mean(torch.stack(dice_scores))
        return loss
