# AI Agent 助手

一个可演示的本地 AI Agent 项目：支持公司知识库问答、数学计算和普通对话。

## 功能

- 知识库检索（`knowledge/*.txt`）
- 计算器工具
- 普通对话
- FastAPI 接口 + 简单网页 UI

## 环境

- Python 3.12
- 依赖见下方安装命令
- 需要 Groq API Key（写入 `.env`）

## 安装

```bash
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn openai python-dotenv sentence-transformers scikit-learn numpy
```

## 配置

在 .env 中设置：

env

```
GROQ_API_KEY=你的密钥
```

## 启动

Bash

```
uvicorn 34_agent_api:app --reload
```

- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 网页界面：[http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)

## 接口示例

POST /agent

JSON

```
{ "question": "WiFi密码是什么？" }
```

返回：

JSON

```
{
  "answer": "...",
  "tool": "search_knowledge",
  "tool_input": "...",
  "tool_result": "..."
}
```

## 知识库

将文本放到 knowledge/ 目录，重启服务后生效。