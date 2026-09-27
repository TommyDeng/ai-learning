import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

MAX_TURNS = 6


def create_client():
    load_dotenv()
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
    )


def create_initial_messages():
    return [{"role": "system", "content": "你是一个友好的AI助手，请用中文回答。"}]


def get_user_input():
    return input("你: ").strip()


def get_ai_response(client, messages):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content


def trim_messages(messages):
    max_length = 1 + MAX_TURNS * 2
    if len(messages) > max_length:
        messages[:] = [messages[0]] + messages[-(max_length - 1) :]
    return messages


def print_history(messages):
    print("当前对话历史：")
    for msg in messages:
        if msg["role"] == "system":
            continue
        print(f"{msg['role']}: {msg['content']}")
    print()


def print_help():
    print("可用命令：")
    print("  退出 / exit / quit  - 结束程序")
    print("  清除 / clear        - 清空对话历史")
    print("  历史 / history      - 查看当前对话历史")
    print("  帮助 / help         - 显示帮助信息\n")


def main():
    client = create_client()
    messages = create_initial_messages()

    print("开始对话吧！输入「帮助」查看可用命令。\n")

    while True:
        user_input = get_user_input()

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ["退出", "exit", "quit"]:
            print("程序已退出，再见！")
            break

        if cmd in ["清除", "clear"]:
            messages = create_initial_messages()
            print("对话已清除。\n")
            continue

        if cmd in ["历史", "history"]:
            print_history(messages)
            continue

        if cmd in ["帮助", "help"]:
            print_help()
            continue

        # 正常对话流程
        messages.append({"role": "user", "content": user_input})

        try:
            ai_reply = get_ai_response(client, messages)
            print("AI:", ai_reply)
            print()

            messages.append({"role": "assistant", "content": ai_reply})
            trim_messages(messages)

        except OpenAIError as e:
            print("调用 AI 出错了：", e)
            # 出错时把刚刚添加的用户消息撤回来，保持历史干净
            messages.pop()
        except Exception as e:  # noqa: BLE001
            print("发生未知错误：", e)
            messages.pop()


if __name__ == "__main__":
    main()
