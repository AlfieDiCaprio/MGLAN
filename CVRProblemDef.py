
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(batch_size, 1, 2)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(batch_size, problem_size, 2)
    # shape: (batch, problem, 2)
    colors_xy = generate_binary_tensor(batch_size, problem_size)

    return depot_xy, node_xy, colors_xy




def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((100 - x, y), dim=2)
    dat3 = torch.cat((x, 100 - y), dim=2)
    dat4 = torch.cat((100 - x, 100 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((100 - y, x), dim=2)
    dat7 = torch.cat((y, 100- x), dim=2)
    dat8 = torch.cat((100 - y, 100 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data





def generate_binary_tensor(num_samples, size):
    # 创建一个随机的one-hot编码张量
    A = torch.zeros(num_samples, size, 4)

    # 随机选择第三个维度上的位置为1
    index = torch.randint(0, 4, (num_samples, size))
    A.scatter_(2, index.unsqueeze(-1), 1)
    return A