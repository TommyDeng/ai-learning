# 1. 写入文件
with open("test.txt", "w", encoding="utf-8") as f:
    f.write("这是我的第一行文字\n")
    f.write("这是第二行文字\n")
    f.write("Python 文件操作练习\n")

print("文件写入完成！")

# 2. 读取文件
print("\n读取文件内容：")
with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)

# 3. 按行读取
print("按行读取：")
with open("test.txt", "r", encoding="utf-8") as f:
    for line in f:
        print("行内容：", line.strip())