import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载一个轻量嵌入模型（首次会下载）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. 准备几句话
sentences = ["我喜欢学习人工智能", "我爱学 AI", "今天天气很好", "机器学习很有趣"]

# 3. 转成向量
embeddings = model.encode(sentences)

print("向量形状:", embeddings.shape)  # (4, 384)
print()

# 4. 计算相似度
sims = cosine_similarity(embeddings)

print("相似度矩阵：")
for i, s in enumerate(sentences):
    print(f"{i}: {s}")
print(sims)
print(np.round(sims, 3))
