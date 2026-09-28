import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI(title="Agent API")
app.mount("/static", StaticFiles(directory="static"), name="static")


# 简单内存会话（重启服务会清空）
sessions: dict[str, list[dict]] = {}
MAX_TURNS = 6


@app.get("/ui")
def ui():
    return FileResponse("static/index.html")


def create_llm_client() -> tuple[OpenAI, str]:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        client = OpenAI(
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        return client, model

    # 默认 groq
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return client, model


client, MODEL_NAME = create_llm_client()
print(f"当前 LLM: provider={os.getenv('LLM_PROVIDER', 'groq')}, model={MODEL_NAME}")


KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents: list[str] = []
doc_embeddings = np.array([])


def load_docs(folder: Path = KNOWLEDGE_DIR) -> list[str]:
    docs = []
    for f in folder.glob("*.txt"):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
    return docs


def rebuild_index():
    global documents, doc_embeddings
    documents = load_docs()
    if documents:
        doc_embeddings = embed_model.encode(documents)
    else:
        doc_embeddings = np.array([])
    return len(documents)


# 启动时构建一次
rebuild_index()


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
    # 去掉常见无关符号：等号、问号、中文问号、空格
    expr = expression.strip()
    expr = expr.replace("=", "").replace("?", "").replace("？", "")
    expr = expr.strip()

    if not expr:
        return "错误：空表达式"
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return f"错误：表达式包含非法字符: {expr}"
    try:
        value = eval(expr, {"__builtins__": {}}, {})
        return str(value)
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


def run_agent(user_question: str, session_id: str = "default") -> dict:
    if session_id not in sessions:
        sessions[session_id] = [
            {
                "role": "system",
                "content": "你是中文助手。需要算数时调用 calculator；需要查公司制度时调用 search_knowledge；其他问题直接回答。",
            }
        ]

    tools_spec = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式，例如 3*(10+2)、1+5-9",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "纯数学表达式，只含数字和 + - * / ( )",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "查询公司知识库，例如年假、报销、WiFi等制度问题",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索用的问题"}
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    messages = sessions[session_id]
    messages.append({"role": "user", "content": user_question})

    first = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools_spec,
        tool_choice="auto",
        temperature=0,
        max_tokens=400,
    )
    msg = first.choices[0].message

    tool_name = "none"
    tool_input = ""
    tool_result = None

    if msg.tool_calls:
        call = msg.tool_calls[0]
        tool_name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except Exception:
            args = {}

        if tool_name == "calculator":
            tool_input = args.get("expression", "")
            tool_result = calculator(tool_input)
        elif tool_name == "search_knowledge":
            tool_input = args.get("query", user_question)
            tool_result = search_knowledge(tool_input)
        else:
            tool_result = f"未知工具: {tool_name}"

        messages.append(msg)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(tool_result),
            }
        )

        second = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        answer = second.choices[0].message.content or str(tool_result)
        messages.append({"role": "assistant", "content": answer})
    else:
        answer = msg.content or ""
        messages.append({"role": "assistant", "content": answer})

    # 简单修剪：保留 system + 最近若干条
    max_len = 1 + MAX_TURNS * 4
    if len(messages) > max_len:
        sessions[session_id] = [messages[0]] + messages[-(max_len - 1) :]
    else:
        sessions[session_id] = messages

    return {
        "answer": answer,
        "tool": tool_name,
        "tool_input": tool_input,
        "tool_result": tool_result,
        "session_id": session_id,
    }


class AgentRequest(BaseModel):
    question: str
    session_id: str = "default"


class AgentResponse(BaseModel):
    answer: str
    tool: str | None = None
    tool_input: str | None = None
    tool_result: str | None = None
    session_id: str = "default"


@app.get("/")
def root():
    return {
        "message": "Agent API running",
        "knowledge_docs": len(documents),
        "tools": list(TOOLS.keys()),
    }


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    try:
        result = run_agent(req.question, req.session_id)
        return AgentResponse(**result)
    except Exception as e:
        return AgentResponse(
            answer=f"服务出错: {e}",
            tool=None,
            tool_input=None,
            tool_result=None,
            session_id=req.session_id,
        )


@app.post("/agent/clear")
def clear_session(session_id: str = "default"):
    sessions.pop(session_id, None)
    return {"message": "cleared", "session_id": session_id}


@app.get("/knowledge/list")
def knowledge_list():
    files = [f.name for f in KNOWLEDGE_DIR.glob("*.txt")]
    return {
        "files": files,
        "chunk_count": len(documents),
        "previews": [d[:40] + ("..." if len(d) > 40 else "") for d in documents],
    }


@app.post("/knowledge/upload")
async def knowledge_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="只支持 .txt 文件")

    content = await file.read()
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件必须是 UTF-8 文本")

    if not text:
        raise HTTPException(status_code=400, detail="文件内容为空")

    # 保存到 knowledge 目录
    save_path = KNOWLEDGE_DIR / Path(file.filename).name
    save_path.write_text(text, encoding="utf-8")

    count = rebuild_index()
    return {
        "message": "上传成功，知识库已重建",
        "filename": save_path.name,
        "chunk_count": count,
    }


@app.post("/knowledge/rebuild")
def knowledge_rebuild():
    count = rebuild_index()
    return {"message": "重建完成", "chunk_count": count}


@app.delete("/knowledge/{filename}")
def knowledge_delete(filename: str):
    # 防止路径穿越
    name = Path(filename).name
    if not name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="只能删除 .txt 文件")

    path = KNOWLEDGE_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    path.unlink()
    count = rebuild_index()
    return {
        "message": f"已删除 {name}",
        "chunk_count": count,
    }
