from losses.dice_focal_loss import DiceFocalLoss


def compute_loss(outputs, labels, args):
    criterion = DiceFocalLoss().to(args.device)
    loss = criterion(outputs, labels)
    return loss
