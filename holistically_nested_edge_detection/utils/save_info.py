import os
import numpy as np
import pandas as pd


# 保存训练过程评价参数信息
def save_train_info(train_result, val_result, args):
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', args.exp, args.mode)  # 训练参数结果文件保存路径
    os.makedirs(result_path, exist_ok=True)  # 如果路径不存在，则创建路径

    # 将train_result列表转换为pandas的DataFrame格式
    data_df = pd.DataFrame(train_result)
    data_df.columns = ['train_loss',
                       'train_dice',
                       'train_precision',
                       'train_recall',
                       'train_auc_pr']

    # 设置DataFrame的索引为从0到args.epochs（不包括args.epochs）的整数序列
    data_df.index = np.arange(0, args.epochs, 1)

    # 创建一个ExcelWriter对象，准备将DataFrame写入Excel文件
    writer = pd.ExcelWriter(os.path.join(result_path, 'train_result.xlsx'))

    # 将DataFrame写入Excel文件，并设置浮点数格式为 '%.6f'（即保留6位小数）
    data_df.to_excel(writer, float_format='%.6f', index=False)

    # 关闭ExcelWriter对象
    writer.close()

    # 同样的步骤，但这次是处理验证结果（val_result）
    data_df_val = pd.DataFrame(val_result)
    data_df_val.columns = ['val_loss',
                           'val_dice',
                           'val_precision',
                           'val_recall',
                           'val_auc_pr']
    data_df_val.index = np.arange(0, args.epochs, 1)
    writer_val = pd.ExcelWriter(os.path.join(result_path, 'val_result.xlsx'))
    data_df_val.to_excel(writer_val, float_format='%.6f', index=False)
    writer_val.close()


# 保存验证过程评价参数信息
def save_valid_info(val_result, args):
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', args.exp, args.mode)  # 训练参数结果文件保存路径
    os.makedirs(result_path, exist_ok=True)  # 如果路径不存在，则创建路径

    # 定义指标名称
    metrics = ['val_loss',
               'val_dice',
               'val_precision',
               'val_recall',
               'val_auc_pr']

    # 创建一个DataFrame
    df = pd.DataFrame({
        'Metric': metrics,
        'Value': val_result
    })

    # 保存为Excel文件
    df.to_excel(os.path.join(result_path, 'val_only_result.xlsx'), float_format='%.6f', index=False)
