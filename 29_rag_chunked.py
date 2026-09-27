import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 60, overlap: int = 10) -> list[str]:
    """按字符长度切分，带一点重叠，避免句子被切断后丢上下文"""
    text = " ".join(text.split())  # 简单压空白
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)
    return chunks


# 1. 加载并切分
raw = load_text("knowledge_long.txt")
documents = split_text(raw, chunk_size=60, overlap=10)
print(f"切分后共 {len(documents)} 个片段：")
for i, d in enumerate(documents):
    print(f"  [{i}] {d}")
print()

# 2. 向量化
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def retrieve(query: str, top_k: int = 3, min_score: float = 0.35):
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, doc_embeddings)[0]
    ranked = np.argsort(sims)[::-1]

    results = []
    for idx in ranked[:top_k]:
        score = float(sims[idx])
        if score >= min_score:
            results.append((documents[idx], score))
    return results


def answer(query: str):
    retrieved = retrieve(query)
    print("检索结果：")
    if not retrieved:
        print("  （无高于阈值的片段）")
        return "资料中没有找到相关信息。"

    for doc, score in retrieved:
        print(f"  [{score:.3f}] {doc}")

    context = "\n".join([f"- {doc}" for doc, _ in retrieved])
    prompt = f"""你是公司助手。请只根据资料回答。
资料没有的信息请回答：资料中AI没有找到相关信息。

资料：
{context}

问题：{query}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "你是严谨的公司知识助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    for q in ["年假满五年有多少天？", "报销被退回怎么办？", "公司有食堂吗？"]:
        print("=" * 60)
        print("问题:", q)
        print("回答:", answer(q))
        print()
