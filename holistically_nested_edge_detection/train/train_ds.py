import os
import torch
from tqdm import tqdm
from torch import optim
from model.hed_ds import create_model_ds
from loss.loss_ds import compute_loss_ds
from utils.data_loader import load_data
from utils.metrics import compute_metrics
from utils.save_info import save_train_info


# 定义训练函数
def train_ds(args):
    # cuDNN设置为使用非确定性算法
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 加载数据
    print('------------------')
    print('加载数据')
    train_loader, val_loader = load_data(args)
    print('数据加载成功')

    # 创建模型
    print('------------------')
    print('创建模型')
    model = create_model_ds(args)
    print('模型创建成功')

    # 初始化优化器
    print('------------------')
    print('初始化优化器')
    optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)  # Adam优化器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.optim_gamma)  # 指数衰减动态学习率
    print('优化器初始化成功')

    # 设置模型保存路径
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp)
    os.makedirs(model_path, exist_ok=True)
    print('------------------')
    print('模型保存路径是：', model_path)

    # 开始训练
    print('------------------')
    print('开始训练')

    # 创建两个列表，存储训练和验证结果
    train_RESULT = []
    val_RESULT = []

    # 遍历训练周期
    for epoch in range(args.epochs):
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print("当前学习率: {:.6f}".format(current_lr))

        model.train()  # 切换到训练模式

        # 定义训练模式各项指标
        train_loss = 0.0
        train_dice = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_auc_pr = 0.0

        # 迭代训练数据集
        for step, data in enumerate(tqdm(train_loader, desc='[Train] Epoch' + str(epoch + 1) + '/' + str(
                args.epochs))):  # 显示进度条，指示当前是训练的哪个轮次（epoch）/总轮次
            inputs, labels = data['x'].to(args.device), data['y'].to(args.device)  # 将训练数据和对应的标签移动到GPU上进行加速计算
            optimizer.zero_grad()  # 梯度清零
            outputs1, outputs2, outputs3, outputs4, outputs5, outputsc = model(inputs)  # 前向传播
            loss = compute_loss_ds(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, labels, args)  # 计算损失
            metrics = compute_metrics(outputsc, labels, args)  # 计算评价指标
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 累加epoch内的指标
            train_loss += loss.item()
            train_dice += metrics['Dice']
            train_precision += metrics['Precision']
            train_recall += metrics['Recall']
            train_auc_pr += metrics['AUC PR']

        # 计算每个指标的均值
        num_batches = len(train_loader)
        m_loss = train_loss / num_batches
        m_dice = train_dice / num_batches
        m_precision = train_precision / num_batches
        m_recall = train_recall / num_batches
        m_auc_pr = train_auc_pr / num_batches

        # 打印该轮次训练信息
        print(
            "第{}轮训练指标结果：".format(epoch + 1),
            " train loss: {:.6f}".format(m_loss),
            " train dice:{:.6f}".format(m_dice),
            " train precision: {:.6f}".format(m_precision),
            " train recall:{:.6f}".format(m_recall),
            " train auc pr:{:.6f}".format(m_auc_pr)
        )

        # 创建训练结果列表，存储每个轮次的结果
        train_result = [
            m_loss,
            m_dice,
            m_precision,
            m_recall,
            m_auc_pr
        ]
        train_RESULT.append(train_result)

        model.eval()  # 切换到评估模式

        # 定义评估模式各项指标
        val_loss = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_auc_pr = 0.0

        # 迭代评估数据集
        with torch.no_grad():  # 停止计算梯度
            for step, data in enumerate(tqdm(val_loader, desc='[VALID] Valid ')):  # 显示进度条，指示当前位于评估阶段
                inputs, labels = data['x'].to(args.device), data['y'].to(args.device)  # 将训练数据和对应的标签移动到GPU上进行加速计算
                outputs1, outputs2, outputs3, outputs4, outputs5, outputsc = model(inputs)  # 前向传播
                loss = compute_loss_ds(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, labels, args)  # 计算损失
                metrics = compute_metrics(outputsc, labels, args)  # 计算评价指标

                # 累加评估模式各项指标
                val_loss += loss.item()
                val_dice += metrics['Dice']
                val_precision += metrics['Precision']
                val_recall += metrics['Recall']
                val_auc_pr += metrics['AUC PR']

        # 计算每个指标的均值
        num_val_batches = len(val_loader)
        m_val_loss = val_loss / num_val_batches
        m_val_dice = val_dice / num_val_batches
        m_val_precision = val_precision / num_val_batches
        m_val_recall = val_recall / num_val_batches
        m_val_auc_pr = val_auc_pr / num_val_batches

        # 打印该轮次验证信息
        print(
            "第{}轮验证指标结果：".format(epoch + 1),
            " val loss: {:.6f}".format(m_val_loss),
            " val dice:{:.6f}".format(m_val_dice),
            " val precision: {:.6f}".format(m_val_precision),
            " val recall:{:.6f}".format(m_val_recall),
            " val auc pr:{:.6f}".format(m_val_auc_pr)
        )

        # 创建验证结果列表，存储每个轮次的结果
        val_result = [
            m_val_loss,
            m_val_dice,
            m_val_precision,
            m_val_recall,
            m_val_auc_pr
        ]
        val_RESULT.append(val_result)

        # 每经过一定的训练周期，保存一个模型检查点
        if (epoch + 1) % args.val_every == 0:
            model_name = 'HEDDS_epoch_{}_dice_{:.6f}.pth'.format(epoch + 1, m_val_dice)
            torch.save(model.state_dict(), os.path.join(model_path, model_name))

        scheduler.step()  # 学习率衰减

    # 打印并保存模型训练信息
    print('------------------')
    print('保存模型训练信息')
    save_train_info(train_RESULT, val_RESULT, args)
    print('------------------')
    print('完成训练')
