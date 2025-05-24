import os
import argparse


# Model training parameters
def add_args():
    parser = argparse.ArgumentParser(description='Single seismic horizon tracking')  # Create argument parser with a description

    # Add command-line arguments
    parser.add_argument('--exp', default='craunet_dicefocal9', type=str, help='Name of each run')  # Name of the experiment

    parser.add_argument('--device', default='cuda:0', type=str, help='Training device')  # Training device, default is 'cuda:0'

    parser.add_argument('--mode', default='pred', choices=['train', 'valid', 'test', 'pred'], type=str,
                        help='Network run mode')  # Four possible modes for the network: 'train', 'valid', 'test', 'pred'

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size during training')  # Batch size for training mode

    parser.add_argument('--batch_size_not_train', default=1, type=int,
                        help='Batch size when not in training mode')  # Batch size when not in training mode

    parser.add_argument('--epochs', default=101, type=int, help='Maximum number of training epochs')  # Maximum number of training epochs

    parser.add_argument('--train_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train'),
                        type=str, help='Dataset directory for training')  # Path to the training dataset

    parser.add_argument('--valid_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'valid'),
                        type=str, help='Dataset directory for validation')  # Path to the validation dataset

    parser.add_argument('--test_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test'),
                        type=str, help='Dataset directory for testing')  # Path to the test dataset

    parser.add_argument('--pred_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pred'),
                        type=str, help='Dataset directory for predictions')  # Path to the prediction dataset

    parser.add_argument('--model', default='CBAMResAttentionUNet', type=str, help='Choose model for training')  # Choose model for training

    parser.add_argument('--in_channels', default=1, type=int, help='Number of input channels')  # Number of input channels

    parser.add_argument('--out_channels', default=7, type=int, help='Number of output channels')  # Number of output channels

    parser.add_argument('--loss_func', default='dicefocal9_loss', type=str, help='Choose loss function')  # Choose loss function

    parser.add_argument('--val_every', default=1, type=int, help='Validation frequency')  # Frequency of validation

    parser.add_argument('--optim_lr', default=1e-4, type=float, help='Optimization learning rate')  # Learning rate for the optimizer

    parser.add_argument('--pretrained_model_name', default='CBAMResAttentionUNet_epoch_101_dice_0.9552_CP.pth',
                        type=str,
                        help='Pretrained model name')  # Name of the pretrained model

    parser.add_argument('--workers', default=10, type=int, help='Number of workers')  # Number of workers

    parser.add_argument('--overlap', default=0.25, type=int, help='Overlap for prediction sliding window')  # Overlap for sliding window during prediction

    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')  # Threshold for classification

    args = parser.parse_args()  # Parse command-line arguments

    print()
    print('>>>============= args ====================<<<')
    print()
    print(args)  # Print command-line arguments
    print()
    print('>>>=======================================<<<')

    return args
