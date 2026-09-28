from openai import OpenAI

# 本地 Ollama 默认地址
client = OpenAI(
    api_key="ollama",  # 任意非空即可，本地不校验
    base_url="http://127.0.0.1:11434/v1",
)

response = client.chat.completions.create(
    model="qwen2.5:3b",  # 与 ollama pull 的名称一致
    messages=[
        {"role": "system", "content": "你是一个友好的中文助手。"},
        {"role": "user", "content": "用两句话说明什么是RAG。"},
    ],
    temperature=0.3,
)

print(response.choices[0].message.content)
