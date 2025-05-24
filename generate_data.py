import torch
import pickle
def get_random_problems(batch_size, problem_size):

    depot_xy = torch.randint(20,81,size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.randint(0,101,size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)
    colors_xy = generate_binary_tensor(batch_size, problem_size)

   

    data={'depot_xy':depot_xy,
          "node_xy":node_xy,
          "colors_xy":colors_xy,
          }
    # shape: (batch, problem)
    return data

def generate_binary_tensor(num_samples, size):
    # 初始生成随机的 0 或 1 张量，形状为 (num_samples, size, 4)
    tensor = torch.randint(0, 2, (num_samples, size, 4))

    # 找到第三个维度上全为 0 的位置
    zero_positions = torch.sum(tensor, dim=2) == 0  # 形状为 (num_samples, size)

    # 检查是否存在全为 0 的位置
    if zero_positions.any():
        # 获取全为 0 的位置的索引
        indices = torch.nonzero(zero_positions)  # 形状为 (num_zero_positions, 2)

        # 对于每个全为 0 的位置，随机选择第三个维度上的一个索引，将其值设为 1
        random_indices = torch.randint(0, 4, (indices.size(0),))

        # 利用高级索引一次性修改张量中对应的位置
        tensor[indices[:, 0], indices[:, 1], random_indices] = 1

    return tensor

data = get_random_problems(1, 20)
with open('50testdata__003.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
data2=data['data_test']
with open('50LKH_testdata__003.pkl', 'wb') as f:
    pickle.dump(data2, f, pickle.HIGHEST_PROTOCOL)

print(data['depot_xy'])