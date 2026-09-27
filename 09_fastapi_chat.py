import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 简单的内存会话存储（重启服务会清空）
sessions: dict[str, list[dict]] = {}
MAX_TURNS = 6


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def trim_messages(messages: list[dict]) -> list[dict]:
    max_length = 1 + MAX_TURNS * 2
    if len(messages) > max_length:
        return [messages[0]] + messages[-(max_length - 1) :]
    return messages


@app.get("/")
def read_root():
    return {"message": "Hello, AI API!"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 获取或初始化会话
    if request.session_id not in sessions:
        sessions[request.session_id] = [
            {"role": "system", "content": "你是一个友好的AI助手，请用中文回答。"}
        ]

    messages = sessions[request.session_id]
    messages.append({"role": "user", "content": request.message})
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
    except OpenAIError as e:
        messages.pop()  # 移除用户消息，避免在会话中留下错误的上下文
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    sessions[request.session_id] = trim_messages(messages)

    return ChatResponse(reply=reply, session_id=request.session_id)


class SessionRequest(BaseModel):
    session_id: str = "default"


@app.post("/chat/clear")
def clear_chat(request: SessionRequest):
    if request.session_id in sessions:
        del sessions[request.session_id]
    return {"message": "Chat history cleared.", "session_id": request.session_id}


@app.get("/chat/history", response_model=list[dict])
def get_chat_history(session_id: str = "default"):
    if session_id in sessions:
        return [m for m in sessions[session_id] if m["role"] != "system"]
    return []
