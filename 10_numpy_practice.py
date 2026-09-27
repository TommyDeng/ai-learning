import numpy as np

# 1. 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])

print("一维数组:", a)
print("二维数组:\n", b)
print("形状:", b.shape)
print("数据类型:", a.dtype)

# 2. 基本运算
print("\n全部 +10:", a + 10)
print("全部 *2:", a * 2)
print("求和:", a.sum())
print("平均值:", a.mean())
print("最大值:", a.max(), a.min())

# 3. 常用创建方式
zeros = np.zeros((2, 3))
ones = np.ones((2, 3))
range_arr = np.arange(0, 10, 2)

print("\n全0:\n", zeros)
print("全1:\n", ones)
print("等差序列:", range_arr)
