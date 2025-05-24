from torch import nn
import torch.nn.functional as F


# 平衡交叉熵
class BalancedBCELoss_DS(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred1, pred2, pred3, pred4, pred5, predc, target):
        # 计算权重
        neg = (1. - target).sum()  # 算有多少个0
        pos = target.sum()  # 算有多少个1
        beta = neg / (neg + pos)  # 0所占的比例
        weight = target * beta + (1. - target) * (1. - beta)  # 创建权重张量

        # 计算loss
        loss1 = F.binary_cross_entropy(pred1, target, reduction=self.reduction, weight=weight)
        loss2 = F.binary_cross_entropy(pred2, target, reduction=self.reduction, weight=weight)
        loss3 = F.binary_cross_entropy(pred3, target, reduction=self.reduction, weight=weight)
        loss4 = F.binary_cross_entropy(pred4, target, reduction=self.reduction, weight=weight)
        loss5 = F.binary_cross_entropy(pred5, target, reduction=self.reduction, weight=weight)
        lossc = F.binary_cross_entropy(predc, target, reduction=self.reduction, weight=weight)

        return loss1 + loss2 + loss3 + loss4 + loss5 + lossc


# 计算损失
def compute_loss_ds(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, labels, args):
    criterion = BalancedBCELoss_DS().to(args.device)
    loss = criterion(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, labels)
    return loss
