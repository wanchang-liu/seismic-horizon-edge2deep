import numpy as np


# torch张量正则化函数
def torch_regularization(input_data):
    mean = input_data.mean()
    std = input_data.std()
    if std == 0:
        standardized_data = input_data
    else:
        standardized_data = (input_data - mean) / std
    return standardized_data


# npy数组正则化函数
def regularization(input_data):
    mean = np.mean(input_data)
    std = np.std(input_data)
    if std == 0:
        standardized_data = input_data
    else:
        standardized_data = (input_data - mean) / std
    return standardized_data
