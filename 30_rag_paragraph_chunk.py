import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def load_paragraphs(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts


documents = load_paragraphs("knowledge_long.txt")
print(f"段落数: {len(documents)}")
for i, d in enumerate(documents):
    print(f"[{i}] {d[:40]}...")
print()

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def retrieve(query: str, top_k: int = 2, min_score: float = 0.3):
    q = embed_model.encode([query])
    sims = cosine_similarity(q, doc_embeddings)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idxs:
        score = float(sims[i])
        if score >= min_score:
            results.append((documents[i], score))
    return results


def answer(query: str):
    retrieved = retrieve(query)
    print("检索:")
    if not retrieved:
        return "资料中没有找到相关信息。"

    for doc, score in retrieved:
        print(f"  [{score:.3f}] {doc}")

    context = "\n".join(f"- {doc}" for doc, _ in retrieved)
    prompt = f"""根据下列资料回答问题。
规则：
1. 优先使用资料中的具体数字和原文信息
2. 只有资料完全未提及时，才说“资料中AI没有找到相关信息”

资料：
{context}

问题：{query}
"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "你是严谨的公司知识助手，回答简练准确。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    for q in ["年假满五年有多少天？", "报销被退回怎么办？", "公司有食堂吗？"]:
        print("=" * 60)
        print("问题:", q)
        print("回答:", answer(q))
        print()
