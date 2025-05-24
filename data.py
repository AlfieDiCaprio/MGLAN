
import torch
def generate_binary_tensor(num_samples, size):
    # 创建一个随机的one-hot编码张量
    A = torch.zeros(num_samples, size, 4)

    # 随机选择第三个维度上的位置为1
    index = torch.randint(0, 4, (num_samples, size))
    A.scatter_(2, index.unsqueeze(-1), 1)
    return A
b=generate_binary_tensor(10, 10)
print(b)
# import pickle
#
# path = r'D:\RL+pathing\ccctsp-POMO\NEW_py_ver\CVRP\POMO\50testdata__004.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
#
# f = open(path, 'rb')
# data = pickle.load(f)
#
# print(data)
def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(batch_size, 1, 2)
    # shape: (batch, 1, 2)

    node_xy = torch.rand(batch_size, problem_size, 2)
    # shape: (batch, problem, 2)
    colors_xy = generate_binary_tensor(batch_size, problem_size)
    # 生成所有样本的 'demand'，形状为 (num_samples, size, 4)，元素为 1 到 9 的浮点数

    return depot_xy, node_xy, colors_xy


a=get_random_problems(10, 10)
print(a)