import os
import torch
import numpy as np
from tqdm import tqdm
from model.hed_ds import create_model_ds
from pred.pred_parts_ds import sliding_window_prediction_ds


# 预测函数
def pred_ds(args):
    print('------------------')
    print("开始预测")

    # 加载预训练模型
    print('------------------')
    print("加载模型")
    model = create_model_ds(args)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp,
                              args.pretrained_model_name)
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    print("成功加载模型")
    model.eval()

    # 创建预测文件存储路径
    pred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', args.exp, args.mode)
    initial_folder = os.path.join(pred_path, 'initial')
    binary_folder = os.path.join(pred_path, 'binary')
    os.makedirs(initial_folder, exist_ok=True)
    os.makedirs(binary_folder, exist_ok=True)

    # 逐个预测所有剖面
    print('---')
    print("开始预测")

    for filename in tqdm(os.listdir(args.pred_path), desc='[Pred] Pred'):
        input_file_path = os.path.join(args.pred_path, filename)
        prediction = sliding_window_prediction_ds(input_file_path, model, args, window_x=128, window_y=128)  # 模型预测
        prediction_binary = (prediction > args.threshold).astype(int)  # 预测二值化标签一维数组

        # 构建预测结果保存路径
        output_file_path = os.path.join(initial_folder, filename)
        output_file_binary_path = os.path.join(binary_folder, filename)

        # 保存预测结果
        np.save(output_file_path, prediction)
        np.save(output_file_binary_path, prediction_binary)
