import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI(title="公司知识库问答")

KNOWLEDGE_DIR = Path("knowledge")


def load_all_documents(folder: Path) -> list[str]:
    docs = []
    if not folder.exists():
        return docs
    for file in folder.glob("*.txt"):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs


documents = load_all_documents(KNOWLEDGE_DIR)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents) if documents else np.array([])

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 2


class AskResponse(BaseModel):
    answer: str
    references: list[str]


def retrieve(query: str, top_k: int = 2, min_score: float = 0.3) -> list[str]:
    if len(documents) == 0:
        return []
    q = embed_model.encode([query])
    sims = cosine_similarity(q, doc_embeddings)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    result = []
    for i in idxs:
        if float(sims[i]) >= min_score:
            result.append(documents[i])
    return result


@app.get("/")
def root():
    return {"message": "Knowledge QA API running", "docs": len(documents)}


@app.get("/docs_count")
def docs_count():
    return {"count": len(documents), "previews": [d[:30] + "..." for d in documents]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    refs = retrieve(req.question, req.top_k)
    if not refs:
        return AskResponse(answer="资料中没有找到相关信息。", references=[])

    context = "\n".join(f"- {r}" for r in refs)
    prompt = f"""根据资料回答问题。资料没有的信息请回答：资料中没有找到相关信息。
优先使用资料中的具体数字和原文。

资料：
{context}

问题：{req.question}
"""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "你是严谨的公司知识助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return AskResponse(answer=resp.choices[0].message.content, references=refs)
