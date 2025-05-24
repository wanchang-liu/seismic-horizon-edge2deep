import os
import time
from pred.pred import pred
from train.train import train
from valid.valid import valid
from config.config import add_args
from config.save_config import save_args_info


def main(args):
    start_time = time.time()  # 记录开始时间

    # 根据模式执行相应操作：训练、验证、测试或预测
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid':
        valid(args)
    elif args.mode == 'pred':
        pred(args)

    end_time = time.time()  # 记录结束时间
    save_args_info(args)  # 保存模型配置文件
    elapsed_time = end_time - start_time  # 计算用时
    elapsed_time_str = f'Total time taken for {args.mode}: {elapsed_time:.2f} seconds'
    print("------------------")
    print(elapsed_time_str)  # 打印用时信息

    # 构建保存目录路径
    save_dir = os.path.join(os.path.dirname(__file__), 'logs', args.exp, args.mode)
    os.makedirs(save_dir, exist_ok=True)

    # 将用时信息保存为txt文件
    time_log_path = os.path.join(save_dir, f'{args.mode}ing_time.txt')
    with open(time_log_path, 'w') as f:
        f.write(elapsed_time_str)


if __name__ == '__main__':
    args = add_args()
    main(args)
