import torch

# 假设原始的张量是原_tensor，长度为原_length
original_tensor = torch.tensor([1, 2, 3, 4, 5])  # 示例原始张量
original_length = original_tensor.size(0)  # 获取原始张量的长度

a=[original_tensor]*2
print(a)
