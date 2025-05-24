import os


# Save model training parameters to config.txt file
def save_args_info(args):
    argsDict = args.__dict__  # Convert arguments to a dictionary
    result_path = os.path.join(os.path.dirname(__file__), args.exp, args.mode)  # Path to save the config file
    os.makedirs(result_path, exist_ok=True)  # Create the path if it doesn't exist

    # Training mode
    if args.mode == 'train':
        with open(os.path.join(result_path, 'config_train.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    # Validation mode
    elif args.mode == 'valid':
        with open(os.path.join(result_path, 'config_valid.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    # Testing mode
    elif args.mode == 'test':
        with open(os.path.join(result_path, 'config_test.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    # Prediction mode
    elif args.mode == 'pred':
        with open(os.path.join(result_path, 'config_pred.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
