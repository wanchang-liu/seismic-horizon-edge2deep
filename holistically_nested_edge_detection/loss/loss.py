from torch import nn
import torch.nn.functional as F


# 平衡交叉熵
class BalancedBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # 计算权重
        neg = (1. - target).sum()  # 算有多少个0
        pos = target.sum()  # 算有多少个1
        beta = neg / (neg + pos)  # 0所占的比例
        weight = target * beta + (1. - target) * (1. - beta)  # 创建权重张量

        # 计算loss
        loss = F.binary_cross_entropy(pred, target, reduction=self.reduction, weight=weight)

        return loss


# 计算损失
def compute_loss(outputs, labels, args):
    criterion = BalancedBCELoss().to(args.device)
    loss = criterion(outputs, labels)
    return loss
