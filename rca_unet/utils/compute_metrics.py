import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, \
    precision_score, jaccard_score, roc_auc_score, average_precision_score


# Compute evaluation metrics (multi-class, using one-hot encoded labels)
def compute_metrics_multiclass(outputs, labels, num_classes):
    # Convert model outputs and labels to NumPy arrays and move them to CPU
    y_pred_prob = outputs.detach().cpu().numpy()  # Predicted probability distribution
    y_true = labels.detach().cpu().numpy()  # True labels (one-hot encoded)

    # Convert predictions and true labels to 1D arrays
    y_pred = y_pred_prob.argmax(axis=1).flatten()  # Predicted class labels (1D array)
    y_true = y_true.argmax(axis=1).flatten()  # Convert one-hot encoded labels to integer labels

    # Compute IoU (multi-class)
    iou = jaccard_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Dice coefficient (F1-Score) (multi-class)
    dice = f1_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Precision (multi-class)
    precision = precision_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Recall (Sensitivity) (multi-class)
    recall = recall_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Matthews Correlation Coefficient (multi-class)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Compute Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Compute Balanced Accuracy (average across all classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Compute AUC-ROC (multi-class)
    auc_roc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

    # Compute AUC-PR (multi-class)
    auc_pr = average_precision_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_acc,
        'AUC ROC': auc_roc,
        'AUC PR': auc_pr
    }


# Compute evaluation metrics (multi-class, including optimized parameters)
def compute_metrics_multiclass_npy(outputs_prob, outputs_binary, labels, num_classes):
    # Convert model outputs and labels to NumPy arrays and move them to CPU
    y_pred_prob = outputs_prob.flatten()  # Predicted probability distribution
    y_pred = outputs_binary.flatten()  # Predicted binary results
    y_true = labels.detach().cpu().numpy().flatten()  # True labels (one-hot encoded)

    # Convert one-hot encoded labels to integer labels
    y_true = y_true.argmax(axis=1)  # Convert one-hot labels to class labels

    # Compute IoU (multi-class)
    iou = jaccard_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Dice coefficient (F1-Score) (multi-class)
    dice = f1_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Precision (multi-class)
    precision = precision_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Recall (Sensitivity) (multi-class)
    recall = recall_score(y_true, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    # Compute Matthews Correlation Coefficient (multi-class)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Compute Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Compute Balanced Accuracy (average across all classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Compute AUC-ROC (multi-class)
    auc_roc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

    # Compute AUC-PR (multi-class)
    auc_pr = average_precision_score(y_true, y_pred_prob, average='macro', multi_class='ovr')

    return {
        'IoU': iou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_acc,
        'AUC ROC': auc_roc,
        'AUC PR': auc_pr
    }
