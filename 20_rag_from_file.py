import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def load_documents(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 按空行切分成段落
    docs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return docs


documents = load_documents("knowledge.txt")
print(f"已加载 {len(documents)} 段文档")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def retrieve(query: str, top_k: int = 2):
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, doc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(documents[i], float(sims[i])) for i in top_idx]


def answer(query: str):
    retrieved = retrieve(query, top_k=2)
    context = "\n".join([f"- {doc}" for doc, _ in retrieved])

    print("检索到：")
    for doc, score in retrieved:
        print(f"  [{score:.3f}] {doc}")

    prompt = f"""你是公司助手。请只根据下面资料回答。
如果资料没有相关信息，说“资料中没有找到相关信息”。

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
    for q in ["试用期多久？", "年假怎么算？", "有班车吗？"]:
        print("=" * 50)
        print("问题:", q)
        print("回答:", answer(q))
        print()
