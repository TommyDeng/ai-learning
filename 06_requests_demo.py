import requests

# 发送一个简单的 GET 请求
response = requests.get("https://httpbin.org/get")

print("状态码：", response.status_code)
print("响应内容类型：", response.headers.get("Content-Type"))
print("\n返回的部分内容：")
print(response.text[:300])  # 只打印前300个字符
