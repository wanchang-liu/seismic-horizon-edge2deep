import os


# 将模型训练参数保存到config.txt文件
def save_args_info(args):
    argsDict = args.__dict__  # 将参数转换为字典
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', args.exp, args.mode)  # 配置文件保存路径
    os.makedirs(result_path, exist_ok=True)  # 如果路径不存在，则创建路径

    # 训练模式
    if args.mode == 'train':
        with open(os.path.join(result_path, 'config_train.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    # 验证模式
    elif args.mode == 'valid':
        with open(os.path.join(result_path, 'config_valid.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')

    # 预测模式
    elif args.mode == 'pred':
        with open(os.path.join(result_path, 'config_pred.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
