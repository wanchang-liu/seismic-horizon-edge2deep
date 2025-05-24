import os
import time
from test.test import test
from train.train import train
from valid.valid import valid
from predict.predict import pred
from configs.config import add_args
from configs.save_config import save_args_info


def main(args):
    start_time = time.time()  # Record start time

    # Perform corresponding operation based on the mode: train, validate, test, or predict
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid':
        valid(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'pred':
        pred(args)

    end_time = time.time()  # Record end time

    save_args_info(args)  # Save model configuration

    elapsed_time = end_time - start_time  # Calculate elapsed time
    elapsed_time_str = f'Total time taken for {args.mode}: {elapsed_time:.2f} seconds'
    print("------------------")
    print(elapsed_time_str)  # Print elapsed time information

    # Build the save directory path
    save_dir = os.path.join('logs', args.exp, args.mode)
    os.makedirs(save_dir, exist_ok=True)

    # Save the elapsed time as a txt file
    time_log_path = os.path.join(save_dir, f'{args.mode}ing_time.txt')
    with open(time_log_path, 'w') as f:
        f.write(elapsed_time_str)


if __name__ == '__main__':
    args = add_args()
    main(args)
