import os
import numpy as np
from matplotlib import pyplot as plt


# 保存验证结果
def save_valid_result(segs, inputs, gts, args):
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', args.exp,
                               args.mode)  # 验证参数结果文件保存路径
    path_original_numpy = os.path.join(result_path, 'numpy', 'original')  # 原始npy数组结果文件保存路径
    path_label_numpy = os.path.join(result_path, 'numpy', 'label')  # 标签npy数组结果文件保存路径
    path_segmentation_numpy_prob = os.path.join(result_path, 'numpy', 'segmentation_prob')  # 预测npy数组结果文件保存路径
    path_segmentation_numpy_binary = os.path.join(result_path, 'numpy', 'segmentation_binary')  # 预测npy数组结果二值化文件保存路径

    path_original_picture = os.path.join(result_path, 'picture', 'original')  # 原始图片结果文件保存路径
    path_label_picture = os.path.join(result_path, 'picture', 'label')  # 标签图片结果文件保存路径
    path_segmentation_picture_prob = os.path.join(result_path, 'picture', 'segmentation_prob')  # 预测图片结果文件保存路径
    path_segmentation_picture_binary = os.path.join(result_path, 'picture', 'segmentation_binary')  # 预测图片二值化结果文件保存路径

    # 如果路径不存在，则创建路径
    os.makedirs(path_original_numpy, exist_ok=True)
    os.makedirs(path_label_numpy, exist_ok=True)
    os.makedirs(path_segmentation_numpy_prob, exist_ok=True)
    os.makedirs(path_segmentation_numpy_binary, exist_ok=True)
    os.makedirs(path_original_picture, exist_ok=True)
    os.makedirs(path_label_picture, exist_ok=True)
    os.makedirs(path_segmentation_picture_prob, exist_ok=True)
    os.makedirs(path_segmentation_picture_binary, exist_ok=True)

    # 逐批次操作
    for i in range(len(inputs)):
        seg = segs[i]  # 模型预测概率
        img = inputs[i]  # 模型原始输入
        gt = gts[i]  # 模型验证标签

        # 移除批次和通道维度
        seg = np.squeeze(seg)
        seg_binary = (seg > args.threshold).astype(int)
        img = np.squeeze(img)
        gt = np.squeeze(gt)

        # 保存npy结果
        np.save(os.path.join(path_original_numpy, str(i + 1) + '.npy'), img)  # 原始结果
        np.save(os.path.join(path_label_numpy, str(i + 1) + '.npy'), gt)  # 标签结果
        np.save(os.path.join(path_segmentation_numpy_prob, str(i + 1) + '.npy'), seg)  # 预测结果
        np.save(os.path.join(path_segmentation_numpy_binary, str(i + 1) + '.npy'), seg_binary)  # 预测结果二值化

        # 使用matplotlib的颜色映射直接保存为图像
        plt.imsave(
            os.path.join(path_original_picture, str(i + 1) + '.png'),
            img.T,
            cmap='seismic',
            format='png'
        )
        plt.imsave(
            os.path.join(path_label_picture, str(i + 1) + '.png'),
            gt.T,
            cmap='binary',
            format='png'
        )
        plt.imsave(
            os.path.join(path_segmentation_picture_prob, str(i + 1) + '.png'),
            seg.T,
            cmap='viridis',
            format='png'
        )
        plt.imsave(
            os.path.join(path_segmentation_picture_binary, str(i + 1) + '.png'),
            seg_binary.T,
            cmap='binary',
            format='png'
        )
