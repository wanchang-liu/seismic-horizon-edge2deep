from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        intersection = (pred * target).sum()  # 交集
        total = (pred + target).sum()  # 总和
        loss = 1. - (2. * intersection + self.epsilon) / (total + self.epsilon)  # Dice Loss
        return loss


# 计算损失
def compute_loss(outputs, labels, args):
    criterion = DiceLoss().to(args.device)
    loss = criterion(outputs, labels)
    return loss


# 计算损失
def compute_loss_ds(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, labels, args):
    criterion = DiceLoss().to(args.device)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)
    loss5 = criterion(outputs5, labels)
    lossc = criterion(outputsc, labels)

    return loss1 + loss2 + loss3 + loss4 + loss5 + lossc
