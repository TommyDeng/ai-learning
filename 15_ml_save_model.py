import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 数据
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

# 2. 训练
model = LogisticRegression()
model.fit(X_train, y_train)

print("准确率:", accuracy_score(y_test, model.predict(X_test)))

# 3. 保存模型
joblib.dump(model, "pass_model.joblib")
print("模型已保存为 pass_model.joblib")
