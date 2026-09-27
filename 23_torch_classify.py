import torch
from torch import nn

# 1. 数据（和之前类似）
X = torch.tensor(
    [
        [1.0, 5.0],
        [2.0, 10.0],
        [3.0, 15.0],
        [4.0, 20.0],
        [5.0, 25.0],
        [6.0, 30.0],
        [7.0, 35.0],
        [8.0, 40.0],
        [9.0, 45.0],
        [10.0, 50.0],
    ],
    dtype=torch.float32,
)

y = torch.tensor(
    [[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
    dtype=torch.float32,
)

# 2. 简单网络：2个输入 → 1个输出（概率）
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 3. 训练
for epoch in range(300):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            acc = ((y_pred > 0.5).float() == y).float().mean().item()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Acc: {acc:.2f}")

# 4. 预测新学生
new_students = torch.tensor(
    [
        [3.0, 12.0],  # 应该偏不及格
        [8.0, 40.0],  # 应该偏及格
    ],
    dtype=torch.float32,
)

with torch.no_grad():
    probs = model(new_students)

print("\n预测结果：")
for i, p in enumerate(probs):
    label = "及格" if p.item() > 0.5 else "不及格"
    print(f"学生{i + 1}: 概率={p.item():.3f} → {label}")
