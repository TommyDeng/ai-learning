import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件里的环境变量
load_dotenv()

# 创建客户端（使用 Groq）
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)

# 调用大模型
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",  # 已改为当前可用的免费模型
    messages=[
        {"role": "system", "content": "你是一个友好的AI助手。"},
        {
            "role": "user",
            "content": "你好！请用中文简单介绍一下你自己，并告诉我今天学习AI的建议。",
        },
    ],
    temperature=0.7,
    max_tokens=500,
)

# 打印结果
print("AI 的回复：")
print(response.choices[0].message.content)
