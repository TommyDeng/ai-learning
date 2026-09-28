# AI Learning Lab

个人 AI 工程学习项目与作品集。

从零搭建可运行的本地 AI 应用，覆盖 **大模型应用、RAG、Agent、机器学习、深度学习、目标检测、OCR、本地模型部署**。

---

## 项目简介

本仓库记录完整学习路径与可运行代码，最终形成一组可演示服务：

| 模块 | 说明 |
|------|------|
| **Agent 知识助手** | 多轮对话、工具调用、知识库问答、Web UI |
| **RAG 检索增强** | Embedding 检索 + 大模型生成 |
| **YOLO 检测服务** | 图片目标检测 API |
| **OCR 识别服务** | 图片文字识别 API |
| **本地 LLM** | Ollama 本地模型，支持与 Groq 云端切换 |

适合作为：

- 学习笔记与代码档案
- 个人 GitHub 作品展示
- 后续继续扩展的基础框架

---

## 功能特性

### 1. AI Agent 知识助手
- 多轮会话记忆（`session_id`）
- Function Calling 工具调用
  - `calculator`：数学计算
  - `search_knowledge`：公司知识库检索
- 知识库管理：上传、列表、删除、热更新
- FastAPI 接口 + 简易 Web 界面

### 2. RAG（检索增强生成）
- 文本向量化（SentenceTransformers）
- 余弦相似度检索
- 可基于本地 `knowledge/*.txt` 回答问题
- 支持资料不足时拒答

### 3. 机器视觉
- YOLOv8 预训练检测（单图 / 批量）
- 置信度过滤
- 训练流程跑通（`best.pt` 加载推理）
- YOLO 检测 API（上传图片返回框和类别）

### 4. OCR
- PaddleOCR 中文识别
- OCR API（上传图片返回逐行文本与置信度）

### 5. 本地大模型
- Ollama 部署与调用
- OpenAI 兼容接口
- 通过环境变量在 **Groq / Ollama** 间切换

---

## 技术栈

- **语言**：Python 3.12
- **Web**：FastAPI、Uvicorn
- **LLM**：Groq API、Ollama（OpenAI 兼容）
- **RAG**：sentence-transformers、scikit-learn
- **机器学习**：NumPy、Pandas、scikit-learn
- **深度学习**：PyTorch
- **视觉**：Ultralytics YOLOv8、OpenCV、PaddleOCR
- **前端**：原生 HTML / CSS / JavaScript

---

## 目录结构

```text
AI-Learning/
├── README.md
├── .gitignore
├── .env.example
├── knowledge/                 # Agent 知识库文本
├── static/
│   └── index.html             # Agent Web UI
├── images/                    # 视觉测试图片
├── yolo_out/                  # YOLO 输出（可忽略）
├── 34_agent_api.py            # Agent + RAG + 知识库管理
├── 42_yolo_api.py             # YOLO 检测 API
├── 44_ocr_api.py              # OCR API
├── 45_ollama_chat.py          # 本地模型最小示例
└── 01_xxx.py ...              # 其余分步练习脚本
```


> 练习文件按序号组织（约 01–46），从 Python 基础到 Agent、视觉、本地模型。

---

## 环境要求

- Windows 10 / 11
- Python 3.12
- 建议使用 venv
- （可选）Ollama 本地模型
- （可选）Groq API Key

---

## 安装

Bash

```
# 1. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖（按需）
pip install fastapi uvicorn openai python-dotenv
pip install numpy pandas scikit-learn sentence-transformers
pip install torch torchvision
pip install ultralytics opencv-python
pip install paddlepaddle paddleocr
```

### 安装 Ollama（本地模型）

1. 下载安装：[https://ollama.com/download](https://ollama.com/download)
2. 拉取模型：

Bash

```
ollama pull qwen2.5:3b
```

---

## 配置

1. 复制环境变量模板：

Bash

```
copy .env.example .env
```

2. 编辑 .env：

env

```
GROQ_API_KEY=your_key_here

# groq 或 ollama
LLM_PROVIDER=groq

GROQ_MODEL=openai/gpt-oss-20b
OLLAMA_MODEL=qwen2.5:3b
```

> 请勿将真实 .env 提交到 GitHub。

---

## 快速启动

### A. Agent 知识助手

Bash

```
uvicorn 34_agent_api:app --reload
```

|入口|地址|
|---|---|
|API 文档|[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)|
|Web UI|[http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)|

**示例请求：**

http

```
POST /agent
Content-Type: application/json

{
  "question": "WiFi密码是什么？",
  "session_id": "demo1"
}
```

**常用接口：**

|方法|路径|说明|
|---|---|---|
|POST|/agent|对话 / 工具调用|
|POST|/agent/clear|清空会话|
|GET|/knowledge/list|知识库列表|
|POST|/knowledge/upload|上传 txt|
|DELETE|/knowledge/{filename}|删除知识文件|

---

### B. YOLO 检测 API

Bash

```
uvicorn 42_yolo_api:app --reload --port 8001
```

- 文档：[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
- POST /detect：上传图片
- GET /result/{filename}：获取画框结果图

---

### C. OCR API

Bash

```
uvicorn 44_ocr_api:app --reload --port 8002
```

- 文档：[http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)
- POST /ocr：上传图片，返回识别文本

**说明：** Windows CPU 下 PaddleOCR 可能遇到 oneDNN 兼容问题。本项目代码中使用：

Python

```
os.environ["FLAGS_enable_pir_api"] = "0"
PaddleOCR(lang="ch", enable_mkldnn=False)
```

进行绕过。

---

### D. 本地模型最小示例

Bash

```
python 45_ollama_chat.py
```

确保 Ollama 已运行，并且已 pull 对应模型。

---

## 学习路径（已完成）

### 基础与工程

1. Python 环境、语法、文件与 HTTP
2. LLM API 调用、多轮对话、记忆修剪
3. FastAPI 服务化

### 数据与经典机器学习

4. NumPy / Pandas
5. scikit-learn 训练、评估、模型保存加载

### 深度学习

6. PyTorch 张量、自动求导、回归/分类
7. CNN 与 MNIST
8. RNN / Self-Attention 直觉

### LLM 应用

9. Embedding 与相似度
10. RAG（命令行 → 文件知识库 → API）
11. Agent 与 Function Calling
12. Web UI、会话记忆、知识库热更新

### 视觉与本地化

13. YOLOv8 检测、批量、训练链路、API
14. PaddleOCR 与 OCR API
15. Ollama 本地模型与云端/本地切换

---

## 已验证结果（节选）

- Agent：可正确在「闲聊 / 计算 / 查知识库」间路由
- 知识库：上传与删除后检索结果即时变化
- YOLO：bus.jpg 可稳定检出 person / bus 等目标
- OCR：中文表格/试题图片可返回多行文本
- Ollama：本地 qwen2.5:3b 可被 Python 正常调用

---

## 安全与隐私

- 不提交 API Key、账号密码、私钥
- .gitignore 已忽略：
    - .env
    - venv/
    - __pycache__/
    - 数据与权重（如 data/、*.pt、*.joblib）
- knowledge/ 仅放可公开示例文本，勿放真实内部资料

---

## 后续扩展建议

1. **PDF 解析入库**：PDF → 文本 → 知识库
2. **向量数据库**：Chroma / FAISS / pgvector
3. **OCR 联动 RAG**：图片识别后自动可检索
4. **自定义 YOLO 类别**：标注数据后微调
5. **部署**：Docker、基础鉴权、日志与限流

---

## 作品说明（可用于简历）

**中文：**  
独立完成本地 AI Agent 知识助手与视觉服务。实现多轮对话、Function Calling、基于 Embedding 的知识库问答，以及知识文件热更新；并完成 YOLOv8 检测 API、PaddleOCR 识别 API，支持 Groq 云端与 Ollama 本地模型切换。

**English：**  
Built a local AI Agent knowledge assistant with multi-turn chat, tool calling, embedding-based RAG, and hot-reloadable knowledge base management. Also implemented YOLOv8 detection API, PaddleOCR API, and OpenAI-compatible switching between Groq cloud and Ollama local models.

---

## License

本仓库用于个人学习与作品展示。  
第三方模型与依赖遵循其各自开源许可证。
