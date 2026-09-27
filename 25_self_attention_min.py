import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 假设一句话有 3 个词，每个词 4 维
x = torch.randn(3, 4)  # (序列长度, 维度)
print("输入 x（3个词）:\n", x)
print()

# 简化：直接用线性层生成 Q, K, V（这里用同一套权重演示）
W_q = torch.randn(4, 4)
W_k = torch.randn(4, 4)
W_v = torch.randn(4, 4)

Q = x @ W_q  # (3, 4)
K = x @ W_k
V = x @ W_v

# 注意力分数：Q 和 K 做点积
scores = Q @ K.T / (4**0.5)  # 缩放，防止数值过大
print("注意力分数 scores:\n", scores)
print()

# Softmax 变成权重（每行和为 1）
weights = F.softmax(scores, dim=-1)
print("注意力权重 weights:\n", weights)
print()

# 用权重对 V 加权求和
output = weights @ V
print("Self-Attention 输出:\n", output)
print()

print("解读：")
print("- weights[i][j] 表示第 i 个词对第 j 个词的关注程度")
print("- 输出第 i 行 = 所有词的 V 按 weights[i] 加权混合")
