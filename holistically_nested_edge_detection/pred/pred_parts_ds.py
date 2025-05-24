import torch
import numpy as np
from utils.standardization import regularization


# 滑动窗口预测函数
def sliding_window_prediction_ds(input_data, model, args, window_x=128, window_y=128):
    input_data = np.load(input_data)  # 加载预测数据
    size_x, size_y = input_data.shape  # 预测数据的尺寸

    # 计算窗口滑动的步长
    step_x = int(window_x * (1 - args.overlap))
    step_y = int(window_y * (1 - args.overlap))

    # 初始化输出和权重地图
    output = np.zeros((1, 1, size_x, size_y), dtype=np.float32)
    weight_map = np.zeros((1, 1, size_x, size_y), dtype=int)

    # 滑动窗口预测
    for i in range(0, size_x, step_x):
        # 计算当前方向的起始和结束位置
        x_end = min(i + window_x, size_x)
        x_start = x_end - window_x

        for j in range(0, size_y, step_y):
            # 计算当前方向的起始和结束位置
            y_end = min(j + window_y, size_y)
            y_start = y_end - window_y

            block = input_data[x_start:x_end, y_start:y_end]  # 切块
            block = block.reshape(1, 1, window_x, window_y)  # 调整为[批次，通道，宽度，高度]的格式
            block_normal = regularization(block)  # 正则化
            input_block = torch.from_numpy(block_normal).to(args.device).float()  # 转换为张量
            _, _, _, _, _, block_prediction = model(input_block)  # 模型预测
            block_prediction = block_prediction.detach().cpu().numpy()  # 预测结果移动到cpu上

            # 更新输出和权重地图
            output[:, :, x_start:x_end, y_start:y_end] += block_prediction
            weight_map[:, :, x_start:x_end, y_start:y_end] += 1

    # 计算最终输出
    output /= weight_map
    output = np.squeeze(output)

    return output
