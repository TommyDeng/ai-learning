# 练习1：定义函数
def greet(name):
    return f"你好，{name}！欢迎开始学习 AI。"

# 调用函数
message = greet("Tommy")
print(message)

# 练习2：带参数的计算函数
def add_numbers(x, y):
    result = x + y
    return result

print("15 + 27 =", add_numbers(15, 27))

# 练习3：条件判断
age = 40

if age < 18:
    print("未成年")
elif age < 60:
    print("成年人")
else:
    print("老年人")

# 练习4：简单循环
print("\n开始循环输出：")
for i in range(1, 6):
    print(f"这是第 {i} 次循环")

# 练习5：遍历列表
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
print("\n我喜欢的水果：")
for fruit in fruits:
    print("-", fruit)