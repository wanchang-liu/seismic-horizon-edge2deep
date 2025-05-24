import os
import torch
from tqdm import tqdm
from utils.data_loader import load_data
from model.hed_ds import create_model_ds
from loss.loss_ds import compute_loss_ds
from utils.metrics import compute_metrics
from utils.save_info import save_valid_info
from utils.save_result import save_valid_result


# 定义验证函数
def valid_ds(args):
    # 加载数据
    print('------------------')
    print('加载数据')
    val_loader = load_data(args)
    print('数据加载成功')

    # 加载模型
    print("------------------")
    print("加载模型")
    model = create_model_ds(args)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_models', args.exp,
                              args.pretrained_model_name)  # 设置模型加载路径
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))  # 从模型路径加载模型
    print("模型加载成功")

    # 开始验证
    print('------------------')
    print("开始验证")

    # 初始化列表，用于存储验证结果
    segs = []
    inputs = []
    gts = []

    # 定义验证模式各项指标
    val_loss = 0.0
    val_dice = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_auc_pr = 0.0

    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_loader, desc='[Valid] Valid')):
            x, y = data['x'].to(args.device), data['y'].to(args.device)  # 将训练数据和对应的标签移动到GPU上进行加速计算
            outputs1, outputs2, outputs3, outputs4, outputs5, outputsc = model(x)  # 前向传播
            loss = compute_loss_ds(outputs1, outputs2, outputs3, outputs4, outputs5, outputsc, y, args)  # 计算损失
            metrics = compute_metrics(outputsc, y, args)  # 计算评价指标

            # 累加评估模式各项指标
            val_loss += loss.item()
            val_dice += metrics['Dice']
            val_precision += metrics['Precision']
            val_recall += metrics['Recall']

            val_auc_pr += metrics['AUC PR']

            # 将结果添加到列表中
            segs.append(outputsc.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

    # 计算每个指标的均值
    num_val_batches = len(val_loader)
    m_val_loss = val_loss / num_val_batches
    m_val_dice = val_dice / num_val_batches
    m_val_precision = val_precision / num_val_batches
    m_val_recall = val_recall / num_val_batches
    m_val_auc_pr = val_auc_pr / num_val_batches

    # 创建验证结果列表，存储验证结果
    val_result = [
        m_val_loss,
        m_val_dice,
        m_val_precision,
        m_val_recall,
        m_val_auc_pr,
    ]

    # 打印平均验证损失和度量指标
    print(
        " val loss: {:.4f}".format(m_val_loss),
        " val dice:{:.4f}".format(m_val_dice),
        " val precision: {:.4f}".format(m_val_precision),
        " val recall:{:.4f}".format(m_val_recall),
        " val auc pr:{:.4f}".format(m_val_auc_pr),
    )

    print("------------------")
    print("保存验证信息")

    # 保存验证信息和结果
    save_valid_info(val_result, args)
    save_valid_result(segs, inputs, gts, args)

    print("------------------")
    print("保存完成！")
