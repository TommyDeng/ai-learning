import torch
from torch import nn

# 假设一句话被编码成 4 个词向量，每个词 3 维
# 实际中这些数字来自 embedding，这里先手写方便理解
sentence = torch.tensor(
    [
        [0.1, 0.2, 0.0],  # 词1
        [0.3, 0.1, 0.4],  # 词2
        [0.2, 0.5, 0.1],  # 词3
        [0.6, 0.1, 0.2],  # 词4
    ],
    dtype=torch.float32,
)  # shape: (4, 3)

# 一个最简单的 RNN 单元
rnn = nn.RNN(input_size=3, hidden_size=4, batch_first=False)

# RNN 期望输入形状: (序列长度, batch, 特征)
x = sentence.unsqueeze(1)  # (4, 1, 3)

output, hidden = rnn(x)

print("输入形状:", x.shape)
print("输入:\n", x.squeeze())
print()

print("每个时间步输出形状:", output.shape)
print("每个时间步输出:\n", output.squeeze())
print()
print("最终隐藏状态形状:", hidden.shape)
print("最终隐藏状态:\n", hidden.squeeze())
print()

print("解释：")
print("- 序列长度 4：一句话有 4 个词")
print("- 每个词进 RNN 后，隐藏状态会更新")
print("- 最终 hidden 可以看作整句话的一个摘要向量")
