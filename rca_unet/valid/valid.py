import os
import torch
from tqdm import tqdm
from utils.data_loader import load_data
from logs.save_info import save_valid_info
from utils.create_model import create_model
from utils.compute_loss import compute_loss
from logs.save_result import save_valid_result
from utils.compute_metrics import compute_metrics


# Define the validation function
def valid(args):
    # Set the computation device (GPU or CPU)
    device = torch.device(args.device)
    print('------------------')
    print('Computation device is:', device)

    # Load the data
    print('------------------')
    print('Loading data')
    val_loader = load_data(args)
    print('Data loaded successfully')

    # Load the model
    print("------------------")
    print("Loading model")
    model = create_model(args)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp,
                              args.pretrained_model_name)  # Set the model load path
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Load model from path
    print("Model loaded successfully")

    # Start validation
    print('------------------')
    print("Starting validation")

    # Initialize lists to store validation results
    segs = []
    inputs = []
    gts = []

    # Define validation mode metrics
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_mcc = 0.0
    val_accuracy = 0.0
    val_balanced_acc = 0.0
    val_auc_roc = 0.0
    val_auc_pr = 0.0

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_loader, desc='[Valid] Valid')):
            x, y = data['x'].to(args.device), data['y'].to(args.device)  # Move validation data and labels to GPU for faster computation
            outputs = model(x)  # Forward pass
            outputs = torch.softmax(outputs, dim=1)  # Softmax activation
            loss = compute_loss(outputs, y, args)  # Compute loss
            metrics = compute_metrics(outputs, y)  # Compute evaluation metrics

            # Accumulate validation metrics
            val_loss += loss.item()
            val_iou += metrics['IoU']
            val_dice += metrics['Dice']
            val_precision += metrics['Precision']
            val_recall += metrics['Recall']
            val_mcc += metrics['MCC']
            val_accuracy += metrics['Accuracy']
            val_balanced_acc += metrics['Balanced Accuracy']
            val_auc_roc += metrics['AUC ROC']
            val_auc_pr += metrics['AUC PR']

            # Add results to lists
            segs.append(outputs.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

    # Compute the mean of each metric
    num_val_batches = len(val_loader)
    m_val_loss = val_loss / num_val_batches
    m_val_iou = val_iou / num_val_batches
    m_val_dice = val_dice / num_val_batches
    m_val_precision = val_precision / num_val_batches
    m_val_recall = val_recall / num_val_batches
    m_val_mcc = val_mcc / num_val_batches
    m_val_accuracy = val_accuracy / num_val_batches
    m_val_balanced_acc = val_balanced_acc / num_val_batches
    m_val_auc_roc = val_auc_roc / num_val_batches
    m_val_auc_pr = val_auc_pr / num_val_batches

    # Print the average validation loss and metrics
    print(
        " val loss: {:.4f}".format(m_val_loss),
        " val iou: {:.4f}".format(m_val_iou),
        " val dice:{:.4f}".format(m_val_dice),
        " val precision: {:.4f}".format(m_val_precision),
        " val recall:{:.4f}".format(m_val_recall),
        " val mcc: {:.4f}".format(m_val_mcc),
        " val accuracy: {:.4f}".format(m_val_accuracy),
        " val balanced acc:{:.4f}".format(m_val_balanced_acc),
        " val auc roc:{:.4f}".format(m_val_auc_roc),
        " val auc pr:{:.4f}".format(m_val_auc_pr)
    )

    # Create validation result list to store validation results
    val_result = [
        m_val_loss,
        m_val_iou,
        m_val_dice,
        m_val_precision,
        m_val_recall,
        m_val_mcc,
        m_val_accuracy,
        m_val_balanced_acc,
        m_val_auc_roc,
        m_val_auc_pr
    ]

    print("------------------")
    print("Saving validation information")

    # Save validation information and results
    save_valid_info(val_result, args)
    save_valid_result(segs, inputs, gts, args)

    print("------------------")
    print("Save completed!")
