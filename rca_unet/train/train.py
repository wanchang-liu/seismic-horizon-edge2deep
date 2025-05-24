import os
import torch
from tqdm import tqdm
import torch.optim as optim
from utils.data_loader import load_data
from logs.save_info import save_train_info
from utils.create_model import create_model
from utils.compute_loss import compute_loss
from utils.compute_metrics import compute_metrics


# Define the training function
def train(args):
    # Set cuDNN to use non-deterministic algorithms
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    # Set the computation device (GPU or CPU)
    device = torch.device(args.device)
    print('------------------')
    print('Computation device is:', device)

    # Load the data
    print('------------------')
    print('Loading data')
    train_loader, val_loader = load_data(args)
    print('Data loaded successfully')

    # Create the model
    print('------------------')
    print('Creating model')
    model = create_model(args)
    print(f'{args.model} model created successfully')

    # Initialize the optimizer
    print('------------------')
    print('Initializing optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)  # Using Adam optimizer
    print('Optimizer initialized successfully')

    # Set the model save path
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp)
    os.makedirs(model_path, exist_ok=True)
    print('------------------')
    print('Model save path is:', model_path)

    # Start training
    print('------------------')
    print('Starting training')

    # Create two lists to store training and validation results; declare variables to store the best IOU results
    train_RESULT = []
    val_RESULT = []

    # Iterate through training epochs
    for epoch in range(args.epochs):
        model.train()  # Switch to training mode

        # Define training mode metrics
        train_loss = 0.0

        # Iterate through the training dataset
        for step, data in enumerate(tqdm(train_loader, desc='[Train] Epoch' + str(epoch + 1) + '/' + str(
                args.epochs))):  # Display progress bar, showing which epoch out of total epochs
            inputs, labels = data['x'].to(args.device), data['y'].to(
                args.device)  # Move training data and corresponding labels to the GPU for accelerated computation
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            outputs = torch.softmax(outputs, dim=1)  # Softmax activation
            loss = compute_loss(outputs, labels, args)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            # Accumulate epoch metrics
            train_loss += loss.item()

        # Compute the mean of each metric
        num_batches = len(train_loader)
        m_loss = train_loss / num_batches

        # Print training information for this epoch
        print(
            "Training metrics for epoch {}:".format(epoch + 1),
            " train loss: {:.6f}".format(m_loss)
        )

        # Create training result list, storing results for each epoch
        train_result = [
            m_loss
        ]
        train_RESULT.append(train_result)

        model.eval()  # Switch to evaluation mode

        # Define evaluation mode metrics
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

        # Iterate through the validation dataset
        with torch.no_grad():  # Stop gradient computation
            for step, data in enumerate(
                    tqdm(val_loader, desc='[VALID] Valid ')):  # Display progress bar, showing evaluation phase
                inputs, labels = data['x'].to(args.device), data['y'].to(
                    args.device)  # Move validation data and corresponding labels to the GPU for accelerated computation
                outputs = model(inputs)  # Forward pass
                outputs = torch.softmax(outputs, dim=1)  # Softmax activation
                loss = compute_loss(outputs, labels, args)  # Compute loss
                metrics = compute_metrics(outputs, labels)  # Compute evaluation metrics

                # Accumulate evaluation metrics
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

        # Print validation information for this epoch
        print(
            "Validation metrics for epoch {}:".format(epoch + 1),
            " val loss: {:.6f}".format(m_val_loss),
            " val iou: {:.6f}".format(m_val_iou),
            " val dice:{:.6f}".format(m_val_dice),
            " val precision: {:.6f}".format(m_val_precision),
            " val recall:{:.6f}".format(m_val_recall),
            " val mcc: {:.6f}".format(m_val_mcc),
            " val accuracy: {:.6f}".format(m_val_accuracy),
            " val balanced acc:{:.6f}".format(m_val_balanced_acc),
            " val auc roc:{:.6f}".format(m_val_auc_roc),
            " val auc pr:{:.6f}".format(m_val_auc_pr)
        )

        # Create validation result list, storing results for each epoch
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
        val_RESULT.append(val_result)

        # Save a model checkpoint every certain number of epochs
        if (epoch + 1) % args.val_every == 0:
            model_name = '{}_epoch_{}_dice_{:.6f}.pth'.format(args.model, epoch + 1, m_val_dice)
            torch.save(model.state_dict(), os.path.join(model_path, model_name))

    # Print and save the training information
    print('------------------')
    print('Saving training information')
    save_train_info(train_RESULT, val_RESULT, args)
    print('------------------')
    print('Training completed')
