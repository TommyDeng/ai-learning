import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

data = {
    "study_hours": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
    ],
    "practice_count": [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        8,
        12,
        18,
        22,
        28,
        33,
        38,
        42,
        48,
        55,
    ],
    "passed": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
}
df = pd.DataFrame(data)

X = df[["study_hours", "practice_count"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=["不及格", "及格"]))


# 设置默认字体为支持中文的黑体 (SimHei) 或 Microsoft YaHei
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签

# 解决保存图像是负号 '-' 显示为方块/警告的问题
plt.rcParams["axes.unicode_minus"] = False

# 真实标签与预测标签示例
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# 1. 计算混淆矩阵矩阵数组
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵:\n", cm)

# 2. 可视化绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["负例", "正例"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print("准确率:", accuracy_score(y_true, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_true, y_pred))
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=["负例", "正例"]))
