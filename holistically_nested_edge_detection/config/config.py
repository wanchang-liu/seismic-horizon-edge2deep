import os
import argparse


# 模型训练参数
def add_args():
    parser = argparse.ArgumentParser(description='holistically nested edge detection')  # 创建解析器对象并添加描述

    # 添加命令行参数
    parser.add_argument('--exp', default='hed', type=str, help='name of each run')  # 训练的名称

    parser.add_argument('--device', default='cuda:0', type=str, help='training device')  # 训练的设备，默认为'cuda:0'

    parser.add_argument('--mode', default='pred', choices=['train', 'valid', 'pred'], type=str,
                        help='network run mode')  # 网络的四种运行模式：'train'（训练模式）、'valid'（验证模式）、'pred'（预测模式）

    parser.add_argument('--batch_size_train', default=100, type=int, help='number of train batch size')  # 训练模式时的批处理大小

    parser.add_argument('--batch_size_valid', default=1, type=int, help='number of valid batch size')  # 非训练模式时的批处理大小

    parser.add_argument('--epochs', default=101, type=int, help='max number of training epochs')  # 最大训练轮数

    parser.add_argument('--train_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train'),
                        type=str, help='dataset directory')  # 训练数据集的存储路径

    parser.add_argument('--valid_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'valid'),
                        type=str, help='dataset directory')  # 验证数据集的存储路径

    parser.add_argument('--pred_path',
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pred'),
                        type=str, help='pred data directory')  # 预测数据集的存储路径

    parser.add_argument('--val_every', default=1, type=int, help='validation frequency')  # 验证频率

    parser.add_argument('--optim_lr', default=1e-3, type=float, help='optimization learning rate')  # 优化器学习率

    parser.add_argument('--optim_gamma', default=0.95, type=float, help='learning rate decay rate')  # 优化器衰减率

    parser.add_argument('--pretrained_model_name', default='HED_epoch_51_dice_0.894474.pth', type=str,
                        help='pretrained model name')  # 预训练模型名称

    parser.add_argument('--workers', default=16, type=int, help='number of workers')  # 工作线程数

    parser.add_argument('--overlap', default=0.25, type=int, help='pred‘s overlap')  # 滑动窗口预测时的重叠度

    parser.add_argument('--threshold', default=0.5, type=float, help='classification threshold')  # 分类阈值

    args = parser.parse_args()  # 解析命令行参数

    print()
    print('>>>============= args ====================<<<')
    print()
    print(args)  # 打印命令行参数
    print()
    print('>>>=======================================<<<')

    return args
