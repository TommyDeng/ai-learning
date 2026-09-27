# 综合练习

# 1. 定义一个函数：判断一个数字是奇数还是偶数
def check_odd_even(number):
    if number % 2 == 0:
        return f"{number} 是偶数"
    else:
        return f"{number} 是奇数"

print(check_odd_even(10))
print(check_odd_even(7))

# 2. 定义一个函数：计算列表的平均值
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    return average

scores = [85, 90, 78, 92, 88]
avg = calculate_average(scores)
print(f"平均分是：{avg}")

# 3. 使用字典存储个人信息
person = {
    "name": "Tommy",
    "age": 40,
    "city": "未知",          # 可以改成你所在的城市
    "skills": ["Java", "Web开发"]
}

print("\n个人信息：")
print(f"姓名：{person['name']}")
print(f"年龄：{person['age']}")
print(f"技能：{person['skills']}")

# 4. 给技能列表添加新内容
person["skills"].append("Python")
print("学习 Python 后的技能：", person["skills"])

# 5. 简单的综合函数
def introduce(person_dict):
    skills_str = "、".join(person_dict["skills"])
    return f"我叫{person_dict['name']}，今年{person_dict['age']}岁，会的技能有：{skills_str}。"

print("\n" + introduce(person))