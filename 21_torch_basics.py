import torch

# 1. 创建张量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print("一维张量:", a)
print("二维张量:\n", b)
print("形状:", b.shape)
print()

# 2. 运算
print("a + 10:", a + 10)
print("a * 2:", a * 2)
print("求和:", a.sum().item())
print()

# 3. 自动求导（深度学习核心）
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x
y.backward()  # 反向传播，计算 dy/dx

print("y =", y.item())
print("dy/dx =", x.grad.item())  # 对 x=2，导数应是 2x+3=7
