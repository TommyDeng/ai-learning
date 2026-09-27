from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 加载数据
iris = load_iris()
X = iris.data  # 特征：花萼长宽、花瓣长宽
y = iris.target  # 标签：花的种类（0/1/2）

print("特征形状:", X.shape)
print("标签形状:", y.shape)
print("类别名称:", iris.target_names)
print()

# 2. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)
print()

# 3. 创建并训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估
acc = accuracy_score(y_test, y_pred)
print("准确率:", acc)

# 6. 看前几个预测结果
print("\n前5个真实标签:", y_test[:5])
print("前5个预测结果:", y_pred[:5])
