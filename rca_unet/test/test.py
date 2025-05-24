import os
import torch
from tqdm import tqdm
from utils.data_loader import load_data
from logs.save_info import save_test_info
from utils.create_model import create_model
from logs.save_result import save_test_result
from test.test_parts import sliding_window_test
from utils.compute_metrics import compute_metrics_npy


# Test function
def test(args):
    print('------------------')
    print("Starting testing")

    # Set computation device (GPU or CPU)
    device = torch.device(args.device)
    print('------------------')
    print('Computation device: ', device)

    # Load data
    print('------------------')
    print('Loading data')
    test_loader = load_data(args)
    print('Data loaded successfully')

    # Load pretrained model
    print('------------------')
    print("Loading model")
    model = create_model(args)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp,
                              args.pretrained_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Model loaded successfully")

    # Start testing
    print('------------------')
    print("Testing started")

    # Initialize lists to store validation results
    segs = []
    segs_opti = []
    inputs = []
    gts = []

    # Define test metrics
    test_iou = 0.0
    test_dice = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_mcc = 0.0
    test_accuracy = 0.0
    test_balanced_acc = 0.0
    test_auc_roc = 0.0
    test_auc_pr = 0.0
    test_iou_opti = 0.0
    test_dice_opti = 0.0
    test_precision_opti = 0.0
    test_recall_opti = 0.0
    test_mcc_opti = 0.0
    test_accuracy_opti = 0.0
    test_balanced_acc_opti = 0.0
    test_auc_roc_opti = 0.0
    test_auc_pr_opti = 0.0

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(test_loader, desc='[Test] Testing')):
            input, label = data['x'].to(args.device), data['y'].to(args.device)  # Move the training data and corresponding labels to GPU for acceleration
            prediction_prob, prediction_binary, prediction_opti = sliding_window_test(input, 512, args.overlap, model,
                                                                                      args)  # Sliding window testing
            metrics = compute_metrics_npy(prediction_prob, prediction_binary, label)
            metrics_opti = compute_metrics_npy(prediction_prob, prediction_opti, label)

            segs.append(prediction_binary)
            segs_opti.append(prediction_opti)
            inputs.append(input.detach().cpu().numpy())
            gts.append(label.detach().cpu().numpy())

            test_iou += metrics['IoU']
            test_dice += metrics['Dice']
            test_precision += metrics['Precision']
            test_recall += metrics['Recall']
            test_mcc += metrics['MCC']
            test_accuracy += metrics['Accuracy']
            test_balanced_acc += metrics['Balanced Accuracy']
            test_auc_roc += metrics['AUC ROC']
            test_auc_pr += metrics['AUC PR']
            test_iou_opti += metrics_opti['IoU']
            test_dice_opti += metrics_opti['Dice']
            test_precision_opti += metrics_opti['Precision']
            test_recall_opti += metrics_opti['Recall']
            test_mcc_opti += metrics_opti['MCC']
            test_accuracy_opti += metrics_opti['Accuracy']
            test_balanced_acc_opti += metrics_opti['Balanced Accuracy']
            test_auc_roc_opti += metrics_opti['AUC ROC']
            test_auc_pr_opti += metrics_opti['AUC PR']

        # Calculate the average of each metric
        num_test_batches = len(test_loader)
        m_test_iou = test_iou / num_test_batches
        m_test_dice = test_dice / num_test_batches
        m_test_precision = test_precision / num_test_batches
        m_test_recall = test_recall / num_test_batches
        m_test_mcc = test_mcc / num_test_batches
        m_test_accuracy = test_accuracy / num_test_batches
        m_test_balanced_acc = test_balanced_acc / num_test_batches
        m_test_auc_roc = test_auc_roc / num_test_batches
        m_test_auc_pr = test_auc_pr / num_test_batches
        m_test_iou_opti = test_iou_opti / num_test_batches
        m_test_dice_opti = test_dice_opti / num_test_batches
        m_test_precision_opti = test_precision_opti / num_test_batches
        m_test_recall_opti = test_recall_opti / num_test_batches
        m_test_mcc_opti = test_mcc_opti / num_test_batches
        m_test_accuracy_opti = test_accuracy_opti / num_test_batches
        m_test_balanced_acc_opti = test_balanced_acc_opti / num_test_batches
        m_test_auc_roc_opti = test_auc_roc_opti / num_test_batches
        m_test_auc_pr_opti = test_auc_pr_opti / num_test_batches

        # Print average test loss and metrics
        print(
            " test IoU: {:.6f}".format(m_test_iou),
            " test Dice: {:.6f}".format(m_test_dice),
            " test Precision: {:.6f}".format(m_test_precision),
            " test Recall: {:.6f}".format(m_test_recall),
            " test MCC: {:.6f}".format(m_test_mcc),
            " test Accuracy: {:.6f}".format(m_test_accuracy),
            " test Balanced Accuracy: {:.6f}".format(m_test_balanced_acc),
            " test AUC ROC: {:.6f}".format(m_test_auc_roc),
            " test AUC PR: {:.6f}".format(m_test_auc_pr),
            " test IoU (Optimized): {:.6f}".format(m_test_iou_opti),
            " test Dice (Optimized): {:.6f}".format(m_test_dice_opti),
            " test Precision (Optimized): {:.6f}".format(m_test_precision_opti),
            " test Recall (Optimized): {:.6f}".format(m_test_recall_opti),
            " test MCC (Optimized): {:.6f}".format(m_test_mcc_opti),
            " test Accuracy (Optimized): {:.6f}".format(m_test_accuracy_opti),
            " test Balanced Accuracy (Optimized): {:.6f}".format(m_test_balanced_acc_opti),
            " test AUC ROC (Optimized): {:.6f}".format(m_test_auc_roc_opti),
            " test AUC PR (Optimized): {:.6f}".format(m_test_auc_pr_opti)
        )
        # Create test result list to store test results
        test_result = [
            m_test_iou,
            m_test_dice,
            m_test_precision,
            m_test_recall,
            m_test_mcc,
            m_test_accuracy,
            m_test_balanced_acc,
            m_test_auc_roc,
            m_test_auc_pr,
            m_test_iou_opti,
            m_test_dice_opti,
            m_test_precision_opti,
            m_test_recall_opti,
            m_test_mcc_opti,
            m_test_accuracy_opti,
            m_test_balanced_acc_opti,
            m_test_auc_roc_opti,
            m_test_auc_pr_opti
        ]

        save_test_info(test_result, args)
        save_test_result(segs, segs_opti, inputs, gts, args)
