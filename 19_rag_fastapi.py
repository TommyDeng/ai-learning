import os

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI()

# 知识库
documents = [
    "公司年假制度：员工入职满一年后，每年享有5天带薪年假。满三年后增加到10天。",
    "报销流程：先在OA系统提交报销单，上传发票，部门经理审批后，财务3个工作日内打款。",
    "WiFi名称是 Company-Office，密码是 Welcome2026。访客网络密码请前台领取。",
    "上班时间是周一到周五 9:00-18:00，中午休息 12:00-13:30。",
]

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


class RagRequest(BaseModel):
    question: str
    top_k: int = 2


class RagResponse(BaseModel):
    answer: str
    retrieved: list[str]


def retrieve(query: str, top_k: int = 2):
    query_vec = embed_model.encode([query])
    sims = cosine_similarity(query_vec, doc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [documents[i] for i in top_idx]


@app.post("/rag/ask", response_model=RagResponse)
def rag_ask(request: RagRequest):
    retrieved_docs = retrieve(request.question, request.top_k)
    context = "\n".join([f"- {d}" for d in retrieved_docs])

    prompt = f"""你是公司助手。请只根据下面提供的资料回答问题。
如果资料里没有相关信息，就说“资料中没有找到相关信息”。

资料：
{context}

问题：{request.question}
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
    answer = response.choices[0].message.content
    return RagResponse(answer=answer, retrieved=retrieved_docs)


@app.get("/")
def root():
    return {"message": "RAG API is running"}
