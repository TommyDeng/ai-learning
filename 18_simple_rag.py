import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# 1. 准备知识库（先用几段假文档）
documents = [
    "公司年假制度：员工入职满一年后，每年享有5天带薪年假。满三年后增加到10天。",
    "报销流程：先在OA系统提交报销单，上传发票，部门经理审批后，财务3个工作日内打款。",
    "WiFi名称是 Company-Office，密码是 Welcome2026。访客网络密码请前台领取。",
    "上班时间是周一到周五 9:00-18:00，中午休息 12:00-13:30。",
]

# 2. 向量化文档
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

# 3. 大模型客户端
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def retrieve(query: str, top_k: int = 2):
    """根据问题检索最相关的文档"""
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, doc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(documents[i], float(sims[i])) for i in top_idx]


def answer(query: str):
    # 检索
    retrieved = retrieve(query, top_k=2)
    context = "\n".join([f"- {doc}" for doc, score in retrieved])

    print("检索到的相关内容：")
    for doc, score in retrieved:
        print(f"  [{score:.3f}] {doc}")
    print()

    # 生成
    prompt = f"""你是公司助手。请只根据下面提供的资料回答问题。
如果资料里没有相关信息，就说“资料中没有找到相关信息”。

资料：
{context}

问题：{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "你是一个严谨的公司知识助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    questions = ["年假有多少天？", "WiFi密码是什么？", "怎么报销？", "公司有健身房吗？"]

    for q in questions:
        print("=" * 50)
        print("问题:", q)
        print("回答:", answer(q))
        print()
