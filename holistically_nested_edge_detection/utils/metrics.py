from sklearn.metrics import recall_score, f1_score, precision_score, average_precision_score


# 计算评价指标
def compute_metrics(outputs, labels, args):
    # 将模型输出和标签转换为NumPy数组，并移动到CPU上
    y_pred_prob = outputs.detach().cpu().numpy().flatten()  # 预测概率标签一维数组
    y_true = labels.detach().cpu().numpy().flatten()  # 真实标签一维数组
    y_pred = (y_pred_prob > args.threshold).astype(int)  # 预测二值化标签一维数组

    # 计算dice系数(F1-Score)
    dice = f1_score(y_true, y_pred, average='binary', zero_division=0)

    # 计算查准率
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)

    # 计算召回率（灵敏度）
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

    # 计算AUC-PR
    auc_pr = average_precision_score(y_true, y_pred_prob)

    return {
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'AUC PR': auc_pr
    }
