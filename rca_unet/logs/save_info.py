import os
import numpy as np
import pandas as pd


# Save training evaluation parameters
def save_train_info(train_result, val_result, args):
    result_path = os.path.join(os.path.dirname(__file__), args.exp, args.mode)  # Path to save the training result file
    os.makedirs(result_path, exist_ok=True)  # Create the path if it doesn't exist

    # Convert the train_result list to a pandas DataFrame
    data_df = pd.DataFrame(train_result)
    data_df.columns = ['train_loss']

    # Set the DataFrame index to an integer sequence from 0 to args.epochs (excluding args.epochs)
    data_df.index = np.arange(0, args.epochs, 1)

    # Create an ExcelWriter object to write the DataFrame to an Excel file
    writer = pd.ExcelWriter(os.path.join(result_path, 'train_result.xlsx'))

    # Write the DataFrame to the Excel file, setting the float format to '%.6f' (6 decimal places)
    data_df.to_excel(writer, float_format='%.6f', index=False)

    # Close the ExcelWriter object
    writer.close()

    # Similarly process the validation results (val_result)
    data_df_val = pd.DataFrame(val_result)
    data_df_val.columns = ['val_loss',
                           'val_iou',
                           'val_dice',
                           'val_precision',
                           'val_recall',
                           'val_mcc',
                           'val_accuracy',
                           'val_balanced_acc',
                           'val_auc_roc',
                           'val_auc_pr']
    data_df_val.index = np.arange(0, args.epochs, 1)
    writer_val = pd.ExcelWriter(os.path.join(result_path, 'val_result.xlsx'))
    data_df_val.to_excel(writer_val, float_format='%.6f', index=False)
    writer_val.close()


# Save validation evaluation parameters
def save_valid_info(val_result, args):
    result_path = os.path.join(os.path.dirname(__file__), args.exp,
                               args.mode)  # Path to save the validation result file
    os.makedirs(result_path, exist_ok=True)  # Create the path if it doesn't exist

    # Define metric names
    metrics = ['val_loss',
               'val_iou',
               'val_dice',
               'val_precision',
               'val_recall',
               'val_mcc',
               'val_accuracy',
               'val_balanced_acc',
               'val_auc_roc',
               'val_auc_pr']

    # Create a DataFrame
    df = pd.DataFrame({
        'Metric': metrics,
        'Value': val_result
    })

    # Save as an Excel file
    df.to_excel(os.path.join(result_path, 'val_only_result.xlsx'), float_format='%.6f', index=False)


# Save test evaluation parameters
def save_test_info(val_result, args):
    result_path = os.path.join(os.path.dirname(__file__), args.exp, args.mode)  # Path to save the test result file
    os.makedirs(result_path, exist_ok=True)  # Create the path if it doesn't exist

    # Define metric names
    metrics = ['test_iou',
               'test_dice',
               'test_precision',
               'test_recall',
               'test_mcc',
               'test_accuracy',
               'test_balanced_acc',
               'test_auc_roc',
               'test_auc_pr',
               'test_iou_opti',
               'test_dice_opti',
               'test_precision_opti',
               'test_recall_opti',
               'test_mcc_opti',
               'test_accuracy_opti',
               'test_balanced_acc_opti',
               'test_auc_roc_opti',
               'test_auc_pr_opti']

    # Create a DataFrame
    df = pd.DataFrame({
        'Metric': metrics,
        'Value': val_result
    })

    # Save as an Excel file
    df.to_excel(os.path.join(result_path, 'test_result.xlsx'), float_format='%.6f', index=False)
