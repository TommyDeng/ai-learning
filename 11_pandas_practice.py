import pandas as pd

# 1. 创建 DataFrame（表格）
data = {
    "姓名": ["张三", "李四", "王五", "赵六"],
    "年龄": [23, 31, 27, 35],
    "城市": ["北京", "上海", "广州", "北京"],
    "分数": [88, 92, 79, 95],
}
df = pd.DataFrame(data)

print("原始表格：")
print(df)
print()

# 2. 基本信息
print("形状（行, 列）:", df.shape)
print("列名:", df.columns.tolist())
print()

# 3. 查看数据
print("前2行：")
print(df.head(2))
print()

print("年龄列：")
print(df["年龄"])
print()

# 4. 简单统计
print("分数平均值:", df["分数"].mean())
print("年龄最大值:", df["年龄"].max())
print()

# 5. 条件筛选
print("分数 >= 90 的人：")
print(df[df["分数"] >= 90])
print()

print("北京的人：")
print(df[df["城市"] == "北京"])
print()

# 6. 新增一列
df["是否及格"] = df["分数"] >= 60
print("新增列后：")
print(df)
