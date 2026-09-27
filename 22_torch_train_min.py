import torch
from torch import nn

# 1. 准备数据：y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. 定义一个最简单的网络：一个线性层
model = nn.Linear(in_features=1, out_features=1)

# 3. 损失函数 + 优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01
)  # （学习率 / Learning Rate）：手动设定的步长（例如 lr=0.01）。

# 4. 训练
for epoch in range(200):
    # 前向传播
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        # print(
        #     f"\n学到的参数: weight={model.weight.item():.3f}, bias={model.bias.item():.3f}"
        # )

# 5. 看学到的参数（理想情况接近 weight=2, bias=1）
w, b = model.weight.item(), model.bias.item()
print(f"\n学到的参数: weight={w:.3f}, bias={b:.3f}")

# 6. 预测
test = torch.tensor([[5.0]])
pred = model(test).item()
print(f"当 x=5 时，预测 y={pred:.3f}（期望约 11）")
