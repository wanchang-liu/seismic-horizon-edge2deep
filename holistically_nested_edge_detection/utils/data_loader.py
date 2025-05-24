import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 地震数据集构建
class SeismicDataset(Dataset):
    # 初始化函数，接受数据集路径path，模式mode（默认为训练模式），以及transform（默认为None，即无数据转换）
    def __init__(self, path, mode='train', transform=None):
        # 参数初始化
        self.path = path
        self.transform = transform
        self.mode = mode
        self.image_list, self.label_list = self.load_data()

    # 获取索引为index的数据项
    def __getitem__(self, index):
        # 加载图像数据
        image = np.load(self.image_list[index])
        label = np.load(self.label_list[index])

        # 将图像转化为（通道，高度，宽度）形式
        image = image.reshape((1, image.shape[0], image.shape[1]))
        label = label.reshape((1, label.shape[0], label.shape[1]))

        x = torch.from_numpy(image)  # 将图像转换为PyTorch张量
        y = torch.from_numpy(label)  # 将标签转换为PyTorch张量

        data = {'x': x.float(), 'y': y.float()}  # 创建数据字典，包含图像和标签

        return data

    # 返回数据集长度的函数
    def __len__(self):
        return len(self.image_list)

    # 加载数据的函数
    def load_data(self):
        img_list = []  # 存储图像路径的列表
        label_list = []  # 存储标签路径的列表
        img_path = os.path.join(self.path, 'seismic')  # 图像路径
        label_path = os.path.join(self.path, 'label')  # 标签路径

        # 遍历图像路径下的所有文件名
        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))  # 将图像文件的完整路径添加到列表中
            label_list.append(os.path.join(label_path, item))  # 将对应的标签文件的完整路径添加到列表中（由于x和y的文件名一样，所以用一步加载进来）

        # 返回图像路径列表和标签路径列表
        return img_list, label_list


# 加载地震数据
def load_data(args):
    # 训练模式
    if args.mode == 'train':
        # 加载训练数据
        train_dataset = SeismicDataset(args.train_path, args.mode, transform=None)
        # train_dataset: 这是要加载的数据集。
        # batch_size: 指定每个批次中的样本数目。
        # shuffle: 如果设置为True，则在每个epoch期间对数据进行洗牌。
        # num_workers: 指定用于数据加载的子进程数目。
        # drop_last: 如果数据集的大小不能被batch size整除，设置为True将丢弃最后一个不完整的批次。
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.workers, drop_last=True)

        print('--- 创建训练数据加载器 ---')
        print(len(train_dataset), ': 训练数据集已创建')
        print(len(train_dataloader), ': 训练数据加载器已创建')

        # 加载验证数据
        valid_dataset = SeismicDataset(args.valid_path, args.mode, transform=None)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.workers, drop_last=True)

        print('--- 创建验证数据加载器 ---')
        print(len(valid_dataset), ': 验证数据集已创建')
        print(len(valid_dataloader), ': 验证数据加载器已创建')

        return train_dataloader, valid_dataloader

    # 验证模式
    elif args.mode == 'valid':
        dataset = SeismicDataset(args.valid_path, args.mode, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_valid, shuffle=True, num_workers=args.workers,
                                drop_last=True)

        print('--- 创建验证数据加载器--- ')
        print(len(dataset), ': 验证数据集已创建')
        print(len(dataloader), ': 验证数据加载器已创建')

        return dataloader
