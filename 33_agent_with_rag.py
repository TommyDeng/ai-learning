import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# ===== 知识库 =====
def load_docs(folder: str = "knowledge") -> list[str]:
    docs = []
    p = Path(folder)
    if not p.exists():
        return docs
    for f in p.glob("*.txt"):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs


documents = load_docs()
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents) if documents else np.array([])


def search_knowledge(query: str) -> str:
    if len(documents) == 0:
        return "知识库为空"
    q = embed_model.encode([query])
    sims = cosine_similarity(q, doc_embeddings)[0]
    idxs = np.argsort(sims)[::-1][:2]
    parts = []
    for i in idxs:
        if float(sims[i]) >= 0.3:
            parts.append(documents[i])
    return "\n".join(parts) if parts else "未找到相关资料"


def calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败: {e}"


TOOLS = {
    "calculator": {
        "description": "计算数学表达式，例如 3*(10+2)",
        "func": calculator,
    },
    "search_knowledge": {
        "description": "查询公司知识库，例如年假、报销、WiFi等制度问题",
        "func": search_knowledge,
    },
}


def run_agent(user_question: str) -> str:
    tool_desc = "\n".join(
        [f"- {name}: {info['description']}" for name, info in TOOLS.items()]
    )

    plan_prompt = f"""你是一个会使用工具的助手。
可用工具：
{tool_desc}

判断是否需要工具，只输出 JSON：
1) 需要计算：{{"tool":"calculator","input":"表达式"}}
2) 需要查公司制度/知识：{{"tool":"search_knowledge","input":"检索问句"}}
3) 不需要工具：{{"tool":"none","input":""}}

用户问题：{user_question}
"""

    plan_resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0,
        max_tokens=200,
    )
    plan_text = plan_resp.choices[0].message.content.strip()
    print("模型规划:", plan_text)

    try:
        start = plan_text.find("{")
        end = plan_text.rfind("}") + 1
        plan = json.loads(plan_text[start:end])
    except Exception:
        return "规划解析失败: " + plan_text

    tool = plan.get("tool", "none")
    tool_input = plan.get("input", "")

    if tool == "none":
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "用中文简洁回答。"},
                {"role": "user", "content": user_question},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content

    if tool in TOOLS:
        result = TOOLS[tool]["func"](tool_input)
        print("工具结果:", result)
        final_prompt = f"""用户问题：{user_question}
工具 {tool} 的结果：
{result}

请用中文给出最终回答。若工具显示未找到资料，请明确说明。"""
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content

    return f"未知工具: {tool}"


if __name__ == "__main__":
    questions = [
        "WiFi密码是什么？",
        "25*4+10等于多少？",
        "今天天气怎么样？",  # 既不该算，也不在知识库
    ]
    for q in questions:
        print("=" * 50)
        print("问题:", q)
        print("回答:", run_agent(q))
        print()
