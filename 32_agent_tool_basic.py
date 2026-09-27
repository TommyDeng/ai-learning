import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def calculator(expression: str) -> str:
    """只允许安全的四则运算"""
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
    }
}


def run_agent(user_question: str) -> str:
    tool_desc = "\n".join(
        [f"- {name}: {info['description']}" for name, info in TOOLS.items()]
    )

    plan_prompt = f"""你是一个会使用工具的助手。
可用工具：
{tool_desc}

请判断用户问题是否需要工具。
如果需要计算，输出 JSON：
{{"tool": "calculator", "input": "表达式"}}

如果不需要工具，输出 JSON：
{{"tool": "none", "input": ""}}

只输出 JSON。不要其他文字。

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

    # 取 JSON
    try:
        start = plan_text.find("{")
        end = plan_text.rfind("}") + 1
        plan = json.loads(plan_text[start:end])
    except Exception:
        return "规划解析失败: " + plan_text

    tool = plan.get("tool", "none")
    tool_input = plan.get("input", "")

    if tool == "none":
        # 直接回答
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
        final_prompt = f"""用户问题：{user_question}
工具 {tool} 的结果：{result}
请用中文给出最终回答。"""
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content

    return f"未知工具: {tool}"


if __name__ == "__main__":
    questions = [
        "3*(12+8) 等于多少？",
        "你好，介绍一下你自己",
    ]
    for q in questions:
        print("=" * 50)
        print("问题:", q)
        print("回答:", run_agent(q))
        print()
